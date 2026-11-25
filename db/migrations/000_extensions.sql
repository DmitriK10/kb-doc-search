-- ============================================================================
-- 000_extensions.sql — расширения и базовые проверки
--
-- Требования: PostgreSQL 16+ (проверено на 18.3), pgvector 0.7+ (halfvec),
-- pg_trgm (trigram-фолбэк на опечатки), btree_gist (EXCLUDE по версиям).
--
-- На KVM-сервере pgvector ставится из репозитория PGDG:
--   apt install postgresql-16-pgvector
-- RAM расширений: 0 (библиотека грузится в адресное пространство сервера,
-- ~5-8 МБ разделяемой памяти на всех клиентов).
-- ============================================================================

\echo 'Проверка версии PostgreSQL...'
DO $$
DECLARE
    v int := current_setting('server_version_num')::int;
BEGIN
    IF v < 130000 THEN
        RAISE EXCEPTION 'Требуется PostgreSQL 13+, найден %', current_setting('server_version');
    END IF;
    IF v < 160000 THEN
        RAISE WARNING 'PostgreSQL % — ниже рекомендуемого 16 (проверялся на 16/18)',
            current_setting('server_version');
    END IF;
END $$;

-- pgvector обязателен: без него нет векторного поиска, а он — половина RRF.
CREATE EXTENSION IF NOT EXISTS vector;

-- pg_trgm: фолбэк на опечатки, включается только если FTS дал ноль строк.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- btree_gist: нужен для EXCLUDE-ограничения «две версии одного документа
-- не могут действовать одновременно» (doc_key WITH =, daterange WITH &&).
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- pgcrypto: только если включён debug-буфер текста запроса (см. 005_debug.sql).
-- Шифрование сырого текста запроса — компромисс для отладки качества;
-- по умолчанию буфер ВЫКЛЮЧЕН (позиция «HMAC без буфера»).
CREATE EXTENSION IF NOT EXISTS pgcrypto;

\echo 'Проверка версии pgvector...'
DO $$
DECLARE
    ext_ver text;
    major   int;
    minor   int;
BEGIN
    SELECT extversion INTO ext_ver FROM pg_extension WHERE extname = 'vector';
    IF ext_ver IS NULL THEN
        RAISE EXCEPTION 'pgvector не установлен';
    END IF;

    -- Сравнение версий по major/minor, а не «магическим числом».
    -- Ошибка прошлой редакции: parts[1]*1000 + parts[2] для '0.8.7' давало
    -- 0*1000 + 8 = 8, что меньше 7000, и миграция падала на актуальной
    -- версии 0.8.7. Числовая упаковка версии — источник таких ошибок;
    -- здесь сравниваем компоненты явно.
    major := split_part(split_part(ext_ver, '-', 1), '.', 1)::int;
    minor := split_part(split_part(ext_ver, '-', 1), '.', 2)::int;

    IF major < 0 OR (major = 0 AND minor < 7) THEN
        RAISE EXCEPTION
            'Требуется pgvector 0.7+ (halfvec, halfvec_cosine_ops), найден %',
            ext_ver;
    END IF;

    RAISE NOTICE 'pgvector % — halfvec доступен', ext_ver;
END $$;