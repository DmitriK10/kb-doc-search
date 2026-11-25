-- ============================================================================
-- 002_lexeme.sql — нормализация, лемматизация, tsvector, агрегаты tsquery
--
-- Порядок файлов задан зависимостями: этот файл НЕ ссылается на таблицы и
-- поэтому грузится первым. Всё, что нужно полнотексту, живёт здесь.
--
-- ПРО ё/е — проверено на живом PostgreSQL 18.3, а не предположено:
--     ts_lexize('russian_stem','отчёт') -> {отчет}
--     ts_lexize('russian_stem','отчет') -> {отчет}
--     to_tsvector('russian','Отчёт о командировке')
--       = to_tsvector('russian','Отчет о командировке')  ->  true
--
-- Следствие: словарь russian УЖЕ унифицирует ё/е. Поэтому
--   * ручная замена ё->е в токенизаторе запроса была не только лишней,
--     но и вредной: она создаёт асимметрию, если применять её только
--     к запросу, а к документу — нет. Из app/search/tsquery.py она убрана;
--   * в kb_norm replace('ё','е') оставлен как страховка для НЕсловарных
--     путей (trigram-сходство, сравнение подстрок по grounding.py), где
--     словарь не участвует. На поисковый tsvector он не влияет, потому что
--     применяется к обеим сторонам одинаково.
-- Менять kb_norm — значит переиндексировать все tsv-колонки (REINDEX),
-- иначе документ и запрос снова разойдутся. Это записано и в ADR.
--
-- RAM: 0 (только функции). Стоимость: kb_norm ~0.02 мс на 1000 символов.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- kb_norm(t) — единственная точка нормализации текста.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION kb_norm(t text)
RETURNS text
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT btrim(regexp_replace(
               replace(lower(coalesce(t, '')), 'ё', 'е'),
               '\s+', ' ', 'g'));
$$;

COMMENT ON FUNCTION kb_norm(text) IS
    'Единая нормализация: lower + ё->е + схлопывание пробелов. Применять '
    'симметрично к документу и к запросу; изменение требует REINDEX tsv.';

-- ----------------------------------------------------------------------------
-- kb_lemma(t) — нормальная форма одного слова через словарь russian_stem.
--   'командировку' -> 'командировк'
--   'и'            -> NULL   (стоп-слово словаря: ts_lexize вернул {})
--   'qwerty123'    -> 'qwerty123' (вне словаря: ts_lexize вернул NULL)
--   '2.3'          -> '2.3'  (не слово — версии фильтруются структурно)
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION kb_lemma(t text)
RETURNS text
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
AS $$
DECLARE
    norm text;
    lex  text[];
BEGIN
    norm := kb_norm(t);
    IF norm = '' THEN
        RETURN NULL;
    END IF;
    -- несколько слов в одном аргументе — не слово, лексему не строим
    IF position(' ' IN norm) > 0 THEN
        RETURN NULL;
    END IF;

    lex := ts_lexize('russian_stem', norm);
    IF lex IS NULL THEN
        RETURN norm;            -- вне словаря: аббревиатура, латиница, номер
    END IF;
    IF cardinality(lex) = 0 THEN
        RETURN NULL;            -- стоп-слово русского словаря
    END IF;
    RETURN lex[1];
END;
$$;

COMMENT ON FUNCTION kb_lemma(text) IS
    'Лексема через russian_stem — тот же словарь, что в kb_tsv. '
    'NULL = стоп-слово или не-слово.';

-- ----------------------------------------------------------------------------
-- kb_lemmas(arr) — пакетная лемматизация массива, стоп-слова отброшены.
--
-- Нужна, потому что в GENERATED ALWAYS AS ... STORED PostgreSQL ЗАПРЕЩАЕТ
-- подзапросы (проверено: «в выражении генерируемого столбца нельзя
-- использовать подзапросы»), а массив лемм хочется хранить рядом с ключом.
-- Immutable-функция в generated-колонке разрешена — это и есть обходной путь.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION kb_lemmas(arr text[])
RETURNS text[]
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT COALESCE(
        array_agg(lem ORDER BY ord) FILTER (WHERE lem IS NOT NULL),
        ARRAY[]::text[]
    )
    FROM (
        SELECT u.ord, kb_lemma(u.x) AS lem
        FROM unnest(arr) WITH ORDINALITY AS u(x, ord)
    ) AS t;
$$;

COMMENT ON FUNCTION kb_lemmas(text[]) IS
    'Лемматизирует массив, отбрасывая стоп-слова. Используется в '
    'generated-колонке словаря синонимов.';

-- ----------------------------------------------------------------------------
-- kb_tsv(text, weight) — tsvector с ТОЙ ЖЕ нормализацией, что у kb_lemma.
--   tsv tsvector GENERATED ALWAYS AS (
--       kb_tsv(section_title, 'A') || kb_tsv(body, 'B')
--   ) STORED
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION kb_tsv(t text, weight text DEFAULT 'B')
RETURNS tsvector
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT setweight(to_tsvector('russian', kb_norm(t)), weight::"char");
$$;

COMMENT ON FUNCTION kb_tsv(text, text) IS
    'tsvector с нормализацией kb_norm; вес A для заголовка, B для тела. '
    'Симметрично с kb_lemma. Изменение kb_norm требует REINDEX.';

-- ----------------------------------------------------------------------------
-- Агрегаты склейки tsquery: OR внутри группы, AND между группами.
-- tsquery_or/tsquery_and — встроенные функции операторов || и &&.
-- ----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'kb_or_agg') THEN
        CREATE AGGREGATE kb_or_agg(tsquery) (
            SFUNC = tsquery_or, STYPE = tsquery, PARALLEL = SAFE
        );
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'kb_and_agg') THEN
        CREATE AGGREGATE kb_and_agg(tsquery) (
            SFUNC = tsquery_and, STYPE = tsquery, PARALLEL = SAFE
        );
    END IF;
END $$;

-- ----------------------------------------------------------------------------
-- Самопроверка (после миграции):
--   SELECT kb_lemma('командировку');                 -- командировк
--   SELECT kb_lemma('и');                             -- NULL
--   SELECT kb_lemma('отчёт') = kb_lemma('отчет');     -- true
--   SELECT kb_tsv('Общие положения', 'A');            -- 'положен':1A
-- ----------------------------------------------------------------------------