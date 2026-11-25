-- ============================================================================
-- 003_synonyms.sql — словарь синонимов и сборка tsquery
--
-- Зависит от 002_lexeme.sql: kb_norm, kb_lemma, kb_lemmas, kb_or_agg/kb_and_agg.
--
-- Решение по матчингу (правка B3 + дефект, найденный первым же тестом):
--   Ключ словаря и токен запроса сравниваются ПО ЛЕКСЕМЕ, а не по
--   поверхностной форме. Иначе «командировку» в запросе не находил запись
--   «командировка», и синонимы молча не работали. Лексема ключа считается
--   генерируемой колонкой через тот же kb_lemma, что и лексема запроса —
--   асимметрия невозможна по построению, а не по договорённости.
--
-- RAM: таблица ~40 строк (единицы КБ).
-- Стоимость расширения: ~0.1–0.3 мс на запрос (2 vCPU).
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Словарь. term и expand хранятся ПОВЕРХНОСТНЫМИ формами — так их читает и
-- правит человек. Лексемы считаются автоматически из 002.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS term_synonyms (
    term    TEXT PRIMARY KEY,
    expand  TEXT[] NOT NULL,
    CONSTRAINT term_synonyms_nonempty CHECK (cardinality(expand) > 0)
);

-- term_lemma: лексема ключа. Именно по ней идёт матчинг запроса.
ALTER TABLE term_synonyms
    ADD COLUMN IF NOT EXISTS term_lemma TEXT
        GENERATED ALWAYS AS (kb_lemma(term)) STORED;

-- expand_lemmas: лексемы вариантов. Считаются immutable-функцией, потому что
-- подзапросы в generated-колонках PostgreSQL запрещает.
ALTER TABLE term_synonyms
    ADD COLUMN IF NOT EXISTS expand_lemmas TEXT[]
        GENERATED ALWAYS AS (kb_lemmas(expand)) STORED;

CREATE INDEX IF NOT EXISTS term_synonyms_lemma
    ON term_synonyms (term_lemma) WHERE term_lemma IS NOT NULL;

COMMENT ON COLUMN term_synonyms.term_lemma IS
    'Лексема ключа. Матчинг запроса идёт по ней, поэтому «командировку» '
    'находит запись «командировка».';
COMMENT ON COLUMN term_synonyms.expand_lemmas IS
    'Лексемы вариантов (генерируемая). Over-stemming здесь безопасен: худший '
    'случай — лишний вариант внутри OR-группы, а AND между группами держит шум. '
    'Не «исправлять» без пересмотра решения по групповому расширению.';

-- Наполнение. Варианты — в базовой форме: лемматизация приведёт их сама.
INSERT INTO term_synonyms (term, expand) VALUES
    ('подписывает',   ARRAY['подписывает','подписывать','подписант','подпись','визирует']),
    ('подписант',     ARRAY['подписант','подписывает','подпись','визирование']),
    ('согласовывает', ARRAY['согласовывает','согласование','согласовать','визирует']),
    ('утверждает',    ARRAY['утверждает','утверждение','утвердить']),
    ('агентский',     ARRAY['агентский','агент','агентирования','агентское']),
    ('договор',       ARRAY['договор','контракт','соглашение','сделка']),
    ('командировка',  ARRAY['командировка','командировочный','поездка','выезд']),
    ('отпуск',        ARRAY['отпуск','отпускной','отдых']),
    ('срок',          ARRAY['срок','период','дней','день','дедлайн']),
    ('ответственный', ARRAY['ответственный','ответственность','отвечает','куратор']),
    ('закупка',       ARRAY['закупка','закупочный','приобретение','тендер']),
    ('контрагент',    ARRAY['контрагент','поставщик','партнер','исполнитель']),
    ('суточные',      ARRAY['суточные','возмещение','расходы','компенсация']),
    ('увольнение',    ARRAY['увольнение','расторжение','уход']),
    ('регламент',     ARRAY['регламент','положение','порядок','правила','инструкция']),
    ('приказ',        ARRAY['приказ','распоряжение','указание']),
    ('доверенность',  ARRAY['доверенность','полномочия','представительство']),
    ('отчет',         ARRAY['отчет','отчёт','отчетность'])
ON CONFLICT (term) DO UPDATE SET expand = EXCLUDED.expand;

-- ----------------------------------------------------------------------------
-- kb_expand_query(tokens) -> (variant, gid)
--   OR внутри gid, AND между gid. Порядок gid = порядок токенов в запросе.
--
-- Синонимы подставляются в ИСХОДНОЙ поверхностной форме (expand), а не в
-- лексемной: to_tsquery нормализует их тем же словарём, что и tsv, поэтому
-- лексема получится такой же. Вариант, совпавший с исходным токеном по
-- лексеме, отбрасывается — иначе группа раздувается дублями.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION kb_expand_query(tokens text[])
RETURNS TABLE(variant text, gid int)
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    WITH numbered AS (
        SELECT t.tok, t.ord::int AS gid, kb_lemma(t.tok) AS lem
        FROM unnest(tokens) WITH ORDINALITY AS t(tok, ord)
    ),
    own AS (
        SELECT n.gid, n.tok AS variant, 0 AS prio
        FROM numbered n
        WHERE n.lem IS NOT NULL
    ),
    syn AS (
        SELECT n.gid, s.expand[i] AS variant, i AS prio
        FROM numbered n
        JOIN term_synonyms s ON s.term_lemma = n.lem
        CROSS JOIN generate_subscripts(s.expand, 1) AS i
        WHERE n.lem IS NOT NULL
          AND kb_lemma(s.expand[i]) IS DISTINCT FROM n.lem
    )
    SELECT v.variant, v.gid
    FROM (SELECT * FROM own UNION ALL SELECT * FROM syn) AS v
    ORDER BY v.gid, v.prio;
$$;

COMMENT ON FUNCTION kb_expand_query(text[]) IS
    'Расширяет токены синонимами в лексемном пространстве. '
    'Шаг 1 (Python): токенизация, стоп-фильтр, потолок групп (К1/К2). '
    'Шаг 2 (здесь): подстановка синонимов. Шаг 3 (kb_tsquery): склейка.';

-- ----------------------------------------------------------------------------
-- kb_tsquery(tokens) — OR внутри группы, AND между группами.
--   NULL, если расширять нечего (все токены — стоп-слова).
--
-- Почему AND между группами: если склеивать всё через OR, частотные слова
-- («документ», «как») утопляют ранжирование шумом, который RRF не дофильтрует.
-- AND по 8 группам — компромисс (правка К2: потолок групп в Python).
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION kb_tsquery(tokens text[])
RETURNS tsquery
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    WITH e AS (SELECT * FROM kb_expand_query(tokens)),
    per_group AS (
        SELECT e.gid, kb_or_agg(to_tsquery('russian', kb_norm(e.variant))) AS q
        FROM e
        GROUP BY e.gid
    )
    SELECT kb_and_agg(q)
    FROM per_group
    WHERE q IS NOT NULL AND q <> ''::tsquery;
$$;

-- ----------------------------------------------------------------------------
-- kb_tsquery_or(tokens) — фолбэк максимального recall: все варианты через OR.
-- Включать ТОЛЬКО если kb_tsquery дал меньше 5 строк (правило из правки B3):
-- иначе частотные слова утопят ранжирование, и RRF не спасёт.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION kb_tsquery_or(tokens text[])
RETURNS tsquery
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    WITH e AS (SELECT DISTINCT variant FROM kb_expand_query(tokens)),
    q AS (
        SELECT to_tsquery('russian', kb_norm(e.variant)) AS t
        FROM e
    )
    SELECT kb_or_agg(t) FROM q WHERE t IS NOT NULL AND t <> ''::tsquery;
$$;

-- ----------------------------------------------------------------------------
-- Самопроверка (после миграции):
--   SELECT kb_lemma('командировку');                   -- командировк
--   SELECT count(*) FROM term_synonyms
--     WHERE term_lemma = kb_lemma('командировку');     -- 1  (лексемный матчинг)
--   SELECT * FROM kb_expand_query(ARRAY['подписывает','договор']);
--   SELECT kb_tsquery(ARRAY['подписывает','договор']); -- (подписыв..) & (договор..)
--   SELECT kb_tsquery(ARRAY['и','как']);               -- NULL
-- ----------------------------------------------------------------------------