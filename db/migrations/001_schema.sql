-- ============================================================================
-- 001_schema.sql — схема документов, чанков, журнала запросов и аналитики
--
-- Порядок применения (важен из-за зависимостей):
--   000_extensions.sql -> 002_lexeme.sql -> 003_synonyms.sql -> 001_schema.sql
-- Файл назван 001 по смысловой нумерации, но грузится последним: он зависит
-- от kb_tsv() из 002.
--
-- Ключевые решения, каждое — следствие конкретного риска:
--
-- 1. halfvec вместо vector. 312 измерений × 4 байта = 1248 Б на вектор.
--    20k чанков = 25 МБ, 160k предложений = 200 МБ. halfvec вдвое меньше
--    (12 МБ / 100 МБ) при потере точности в пределах шума tiny-модели.
--    ВАЖНО: оператор и опкласс обязаны быть halfvec-версиями
--    (halfvec_cosine_ops, <=>), иначе PostgreSQL ответит
--    «operator does not exist: halfvec <=> vector».
--
-- 2. chunks.is_active — денормализованный флаг актуальности. Фильтр «только
--    действующая версия» НЕЛЬЗЯ делать через JOIN с documents внутри
--    ORDER BY embedding <=>: JOIN убивает HNSW, и поиск деградирует
--    с 3 мс до 40 мс. Денормализация решает это.
--
-- 3. documents_no_overlap — две версии одного документа не могут действовать
--    одновременно. Защита от главной боли («получили устаревшую информацию»)
--    на уровне БД, а не на уровне кода.
--
-- 4. query_logs без сырого текста. Храним HMAC, а не sha256: короткий запрос
--    «кто подписывает» перебирается по словарю за минуты, HMAC с серверным
--    pepper — нет. Эмбеддинг необратим.
--
-- RAM: схема сама не потребляет. При shared_buffers=256MB и 20k чанках
-- рабочий набор (активные чанки + индексы) ~120–180 МБ, то есть целиком в кэше.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Документы: версионность и актуальность
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id             BIGSERIAL PRIMARY KEY,
    -- Стабильный ключ семейства БЕЗ версии: 'reg_dogovora'. Версия сюда не
    -- входит, иначе v2.3 и v3.1 станут разными документами и фильтр
    -- актуальности перестанет работать.
    doc_key        TEXT NOT NULL,
    title          TEXT NOT NULL,
    version        TEXT NOT NULL,
    doc_type       TEXT NOT NULL DEFAULT 'regulation'
                   CHECK (doc_type IN ('regulation','order','kb','policy','template','other')),
    owner_unit     TEXT,
    owner_role     TEXT,
    owner_contact  TEXT,
    source_path    TEXT NOT NULL,
    -- идемпотентность переиндексации: не изменился — пропускаем
    content_sha256 CHAR(64),
    valid_from     DATE NOT NULL,
    valid_to       DATE,
    is_active      BOOLEAN NOT NULL DEFAULT TRUE,
    -- ACL. Значение {all} = доступен всем (базовый билд). Фильтрация идёт
    -- ПОСЛЕ RRF, потому что фильтр внутри векторного поиска убивает индекс.
    access_groups  TEXT[] NOT NULL DEFAULT ARRAY['all']::TEXT[],
    indexed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_key, version),
    -- Две одновременно действующие версии одного документа запрещены.
    -- Требует btree_gist для GiST по TEXT и daterange.
    CONSTRAINT documents_no_overlap EXCLUDE USING gist (
        doc_key WITH =,
        daterange(valid_from, COALESCE(valid_to, 'infinity'::date), '[)') WITH &&
    )
);

CREATE INDEX IF NOT EXISTS documents_active
    ON documents (doc_key) WHERE is_active;
CREATE INDEX IF NOT EXISTS documents_access
    ON documents USING gin (access_groups);

COMMENT ON COLUMN documents.doc_key IS
    'Стабильный ключ семейства без версии. По нему работает фильтр '
    'актуальности: v2.3 и v3.1 — один doc_key.';
COMMENT ON COLUMN documents.access_groups IS
    'ACL. {all} = доступен всем (базовый билд). Фильтрация после RRF: '
    'фильтр внутри векторного поиска убивает HNSW.';

-- ----------------------------------------------------------------------------
-- Чанки
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id             BIGSERIAL PRIMARY KEY,
    document_id    BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal        INT NOT NULL,
    section_path   TEXT NOT NULL,        -- '4.2' | '5#т1.3'
    section_title  TEXT,
    h1_path        TEXT,                 -- номер РАЗДЕЛА (не пункта)
    -- ДОСЛОВНЫЙ текст: именно он показывается пользователю как цитата
    body           TEXT NOT NULL,
    -- текст для эмбеддинга: документ + раздел + тело
    embed_text     TEXT NOT NULL,
    chunk_type     TEXT NOT NULL DEFAULT 'item'
                   CHECK (chunk_type IN ('item','table_row')),
    char_len       INT NOT NULL,
    table_index    INT,
    row_index      INT,
    -- для склеенных/нарезанных: ordinal родительского куска (правка B5)
    parent_ordinal INT,
    page           INT,
    is_active      BOOLEAN NOT NULL DEFAULT TRUE,
    embedding      halfvec(312) NOT NULL,
    -- tsv через kb_tsv: та же нормализация, что и на стороне запроса
    tsv            tsvector GENERATED ALWAYS AS (
                       kb_tsv(section_title, 'A') || kb_tsv(body, 'B')
                   ) STORED,
    UNIQUE (document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks (document_id);
-- Векторный индекс создаётся отдельно (004_indexes.sql): до 15k чанков
-- последовательный скан дешевле по RAM и времени сборки.

COMMENT ON COLUMN chunks.body IS
    'Дословный текст для показа. Не нормализуется и не сокращается: '
    'цитата должна быть проверяема человеком.';
COMMENT ON COLUMN chunks.is_active IS
    'Денормализован из documents.is_active, чтобы фильтр актуальности не '
    'требовал JOIN и не убивал HNSW. Обновляется kb_refresh_activity().';

-- ----------------------------------------------------------------------------
-- Предложения чанка: реранк без вызова модели на запросе
--
-- Почему отдельная таблица: реранк по предвычисленным векторам — это matmul
-- 200×312, ~10 мс. Вызов модели на 8 предложениях × 10 чанков — ~150 мс
-- и +200 МБ RSS. На 2 ГБ RAM разница решающая.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunk_sentences (
    chunk_id   BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    seq        SMALLINT NOT NULL,
    text       TEXT NOT NULL,
    embedding  halfvec(312) NOT NULL,
    PRIMARY KEY (chunk_id, seq)
);

COMMENT ON TABLE chunk_sentences IS
    'Векторы предложений, предвычисленные на индексации. Реранк на запросе — '
    'чистый numpy без модели. halfvec: 200 МБ -> 100 МБ на 160k предложений.';

-- ----------------------------------------------------------------------------
-- Журнал запросов. Сырой текст НЕ хранится.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS query_logs (
    id             BIGSERIAL PRIMARY KEY,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- HMAC-SHA256(pepper, normalized_query). Не sha256: короткие запросы
    -- восстанавливаются по словарю за минуты, HMAC с секретом — нет.
    query_hmac     CHAR(64) NOT NULL,
    query_len      SMALLINT NOT NULL,
    embedding      halfvec(312) NOT NULL,
    -- 'stale' — ответ по действующей версии с предупреждением о недействительной
    response_type  TEXT NOT NULL
                   CHECK (response_type IN ('success','stale','refusal')),
    refusal_reason TEXT
                   CHECK (refusal_reason IS NULL OR refusal_reason IN
                          ('no_match','low_confidence','ambiguous','stale_only')),
    confidence     REAL NOT NULL,
    top_score      REAL,
    sources        JSONB NOT NULL DEFAULT '[]'::jsonb,
    latency_ms     INT NOT NULL,
    cache_hit      BOOLEAN NOT NULL DEFAULT FALSE,
    user_id_hash   CHAR(64) NOT NULL,
    user_feedback  SMALLINT CHECK (user_feedback IN (-1, 0, 1)),
    feedback_at    TIMESTAMPTZ,
    cluster_id     BIGINT,
    -- TRUE для демо-истории. KPI считается только по is_simulated=false,
    -- иначе на дашборде смешаны измерение и демонстрация (правка A1).
    is_simulated   BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS query_logs_created ON query_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS query_logs_refusal
    ON query_logs (created_at) WHERE response_type = 'refusal';
CREATE INDEX IF NOT EXISTS query_logs_cluster
    ON query_logs (cluster_id) WHERE cluster_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS query_logs_real
    ON query_logs (created_at DESC) WHERE is_simulated = FALSE;

COMMENT ON COLUMN query_logs.is_simulated IS
    'TRUE для демо-истории. Deflection/Refusal/TimeSaved считаются по '
    'is_simulated=false; синтетика показывается отдельной панелью.';

-- ----------------------------------------------------------------------------
-- Кластеры «белых пятен»
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cluster_runs (
    run_id      BIGSERIAL PRIMARY KEY,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    window_from TIMESTAMPTZ NOT NULL,
    k           INT NOT NULL,
    n_vectors   INT NOT NULL
);

CREATE TABLE IF NOT EXISTS query_clusters (
    id             BIGSERIAL PRIMARY KEY,
    run_id         BIGINT NOT NULL REFERENCES cluster_runs(run_id) ON DELETE CASCADE,
    -- стабильный ключ между прогонами: центроиды сопоставляются по косинусу.
    -- Без него Grafana каждый час показывает «новые» красные зоны, и владельцы
    -- перестают реагировать на алерты.
    stable_key     TEXT NOT NULL,
    centroid       halfvec(312) NOT NULL,
    size           INT NOT NULL,
    refusal_share  REAL NOT NULL,
    useless_share  REAL NOT NULL,
    zone           TEXT NOT NULL CHECK (zone IN ('green','yellow','red')),
    label          TEXT NOT NULL,
    nearest_chunks JSONB NOT NULL DEFAULT '[]'::jsonb,
    owner_role     TEXT,
    status         TEXT NOT NULL DEFAULT 'new'
                   CHECK (status IN ('new','notified','resolved','ignored'))
);

CREATE INDEX IF NOT EXISTS query_clusters_run ON query_clusters (run_id);

COMMENT ON COLUMN query_clusters.label IS
    'Метка берётся из ПРОСТРАНСТВА ДОКУМЕНТОВ (ближайшие чанки к центроиду), '
    'а не из текста запроса: сырой текст не хранится (152-ФЗ).';

-- ----------------------------------------------------------------------------
-- Владельцы тем: используются в отказе («обратитесь к ...») и в алертах
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS knowledge_owners (
    topic_key  TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    role       TEXT NOT NULL,
    contact    TEXT NOT NULL,
    embedding  halfvec(312),
    doc_keys   TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[]
);

-- ----------------------------------------------------------------------------
-- Алерты владельцам. Дедупликация обязательна: почасовой крон иначе пришлёт
-- одно и то же уведомление 24 раза в сутки.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS knowledge_alerts (
    id          BIGSERIAL PRIMARY KEY,
    cluster_id  BIGINT REFERENCES query_clusters(id) ON DELETE SET NULL,
    reason      TEXT NOT NULL
                CHECK (reason IN ('refusals_3','useless_3','quality_drop')),
    payload     JSONB NOT NULL,
    created_day DATE NOT NULL DEFAULT current_date,
    sent_at     TIMESTAMPTZ,
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','sent','failed'))
);

CREATE UNIQUE INDEX IF NOT EXISTS knowledge_alerts_dedup
    ON knowledge_alerts (cluster_id, reason, created_day);

-- ----------------------------------------------------------------------------
-- Кэш ответов. Ключ ОБЯЗАН включать подпись ACL.
--
-- Дефект, найденный в раунде 3: кэш по одному query_hmac отдаёт зарплатное
-- положение сотруднику без прав, если до него тот же вопрос задал HR.
-- Поэтому acl_signature входит в cache_key, и PK составной.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS answer_cache (
    cache_key        CHAR(64) NOT NULL,
    -- Колонка нужна для АНАЛИТИКИ (попадания по группам доступа). Защиту
    -- обеспечивает cache_key, в который подпись уже входит.
    acl_signature    TEXT NOT NULL,
    payload          JSONB NOT NULL,
    index_generation INT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at       TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (cache_key, acl_signature)
);

CREATE INDEX IF NOT EXISTS answer_cache_expiry ON answer_cache (expires_at);

COMMENT ON COLUMN answer_cache.acl_signature IS
    'Дублирует информацию из cache_key (PK включает подпись) — оставлено '
    'только для аналитики попаданий по группам доступа.';

-- ----------------------------------------------------------------------------
-- Изменения между версиями документа (правка B2)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_changes (
    id           BIGSERIAL PRIMARY KEY,
    doc_key      TEXT NOT NULL,
    from_version TEXT NOT NULL,
    to_version   TEXT NOT NULL,
    section_path TEXT NOT NULL,
    change_type  TEXT NOT NULL
                 CHECK (change_type IN ('added','removed','modified')),
    similarity   REAL,
    computed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_key, from_version, to_version, section_path)
);

-- ----------------------------------------------------------------------------
-- Пересчёт актуальности. Вызывается таймером ежедневно в 00:05.
--
-- Почему функция, а не триггер: актуальность зависит от ТЕКУЩЕЙ ДАТЫ, а не от
-- изменения строки. Триггер не сработает в тот день, когда документ устарел
-- сам по себе.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION kb_refresh_activity()
RETURNS TABLE(docs_changed BIGINT, chunks_changed BIGINT)
LANGUAGE plpgsql
AS $$
DECLARE
    d BIGINT := 0;
    c BIGINT := 0;
BEGIN
    WITH upd AS (
        UPDATE documents SET is_active = (
            valid_from <= current_date
            AND (valid_to IS NULL OR valid_to > current_date)
        )
        WHERE is_active IS DISTINCT FROM (
            valid_from <= current_date
            AND (valid_to IS NULL OR valid_to > current_date)
        )
        RETURNING 1
    )
    SELECT count(*) INTO d FROM upd;

    WITH upd2 AS (
        UPDATE chunks ch SET is_active = doc.is_active
        FROM documents doc
        WHERE doc.id = ch.document_id
          AND ch.is_active IS DISTINCT FROM doc.is_active
        RETURNING 1
    )
    SELECT count(*) INTO c FROM upd2;

    RETURN QUERY SELECT d, c;
END;
$$;

COMMENT ON FUNCTION kb_refresh_activity() IS
    'Пересчитывает is_active у документов и чанков по текущей дате. '
    'Денормализация нужна, чтобы фильтр актуальности не убивал HNSW.';