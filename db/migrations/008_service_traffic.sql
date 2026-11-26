-- ============================================================================
-- 008_service_traffic.sql — служебный трафик не попадает в панель
--
-- ПРОБЛЕМА. Панель считает KPI по is_simulated = FALSE, то есть «реальные
-- запросы». Но туда попадали и МОИ проверки: прогоны eval, свип порогов,
-- ручные curl-запросы. К вечеру 26.11 из 845 «реальных» запросов 688 были
-- моими тестами.
--
-- Это прямая угроза достоверности: на слайде «845 запросов за день» при
-- 10 тестировщиках жюри спросит, откуда столько, и ответ будет неудобным.
-- Честная цифра — только запросы респондентов.
--
-- РЕШЕНИЕ. Служебные идентификаторы помечаются is_service = TRUE и
-- исключаются из панели. Записи не удаляем: журнал нужен для отладки,
-- но KPI считается по людям.
--
-- ВАЖНО ПРО ЧЕСТНОСТЬ. Респонденты приходят с анонимным session-id
-- (localStorage, случайный UUID). Он не содержит личных данных и не
-- связывает запросы с человеком — только различает сессии, чтобы панель
-- могла показать «5 респондентов» вместо одного ведра.
--
-- RAM: 0 (одна булева колонка). Стоимость: 0.
-- ============================================================================

ALTER TABLE query_logs
    ADD COLUMN IF NOT EXISTS is_service BOOLEAN NOT NULL DEFAULT FALSE;

COMMENT ON COLUMN query_logs.is_service IS
    'Служебный трафик: прогоны eval, свип порогов, ручные проверки. '
    'Исключается из панели KPI: это не запросы сотрудников. Журнал '
    'сохраняется для отладки, но в «реальные запросы» не входит.';

-- Индекс под панель: реальные не-служебные запросы.
CREATE INDEX IF NOT EXISTS query_logs_real
    ON query_logs (created_at DESC)
    WHERE NOT is_simulated AND NOT is_service;

-- ----------------------------------------------------------------------------
-- Разметка уже накопленного служебного трафика.
--
-- Идентификаторы известны из кода: eval.py использует user_id="eval",
-- sweep_thresholds.py — "sweep", ручные проверки — "t". Считаем хэши
-- здесь же, чтобы не хардкодить их в миграции.
-- ----------------------------------------------------------------------------
UPDATE query_logs
SET is_service = TRUE
WHERE NOT is_simulated
  AND user_id_hash IN (
      encode(sha256('t'::bytea), 'hex'),
      encode(sha256('eval'::bytea), 'hex'),
      encode(sha256('sweep'::bytea), 'hex'),
      encode(sha256('test'::bytea), 'hex'),
      encode(sha256('u'::bytea), 'hex'),
      encode(sha256('u1'::bytea), 'hex'),
      encode(sha256('cm'::bytea), 'hex'),
      encode(sha256('cm-service'::bytea), 'hex'),
      encode(sha256('matrix'::bytea), 'hex')
  );

-- ВАЖНО: 'web-demo' НЕ помечается служебным. До появления session-id
-- все веб-пользователи шли под этим идентификатором, и их запросы —
-- это НАСТОЯЩИЕ сотрудники. Пометить их служебными значило бы выбросить
-- реальный трафик. Минус: различить, сколько было людей, нельзя, поэтому
-- такие записи идут как «сессия unknown».

-- Возврат: если миграция уже применялась, снимаем пометку с web-demo.
UPDATE query_logs
SET is_service = FALSE
WHERE is_service
  AND NOT is_simulated
  AND user_id_hash = encode(sha256('web-demo'::bytea), 'hex');


-- ----------------------------------------------------------------------------
-- Ночные прогоны: 342 запроса от 14 «сессий» за одну минуту (00:50-00:51).
-- Это автоматические тесты с произвольными user_id — пометить их списком
-- невозможно. Признак объективный: живой человек не делает 33 запроса
-- за минуту, и 14 разных сессий не появляются одновременно.
--
-- Всё, что пришло до 10:00 (начало рассылки), считаем служебным.
-- ----------------------------------------------------------------------------
UPDATE query_logs
SET is_service = TRUE
WHERE NOT is_simulated
  AND created_at < '2026-11-26 10:00:00+03'
  AND user_id_hash <> encode(sha256('web-demo'::bytea), 'hex');

-- ----------------------------------------------------------------------------
-- Тот же признак для любых будущих пачек: сессия, сделавшая больше 40
-- запросов за минуту, — это скрипт, а не человек. ЧЕЛОВЕЧЕСКИЙ темп:
-- 3-5 вопросов в минуту максимум при активном тестировании.
-- ----------------------------------------------------------------------------
WITH burst AS (
    SELECT user_id_hash, date_trunc('minute', created_at) AS m, count(*) AS n
    FROM query_logs
    WHERE NOT is_simulated AND NOT is_service
    GROUP BY 1, 2
    HAVING count(*) > 40
)
UPDATE query_logs q
SET is_service = TRUE
FROM burst b
WHERE NOT q.is_simulated
  AND q.user_id_hash = b.user_id_hash
  AND date_trunc('minute', q.created_at) = b.m;

-- ----------------------------------------------------------------------------
-- Представление для панели: только запросы людей.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_real_queries AS
SELECT *
FROM query_logs
WHERE NOT is_simulated AND NOT is_service;

COMMENT ON VIEW v_real_queries IS
    'Запросы реальных сотрудников: без синтетики и без служебных прогонов. '
    'Это источник KPI на панели.';

-- ----------------------------------------------------------------------------
-- Проверка: сколько осталось реального.
-- ----------------------------------------------------------------------------
DO $$
DECLARE
    real_n INT;
    serv_n INT;
    users_n INT;
BEGIN
    SELECT count(*) INTO real_n FROM query_logs
    WHERE NOT is_simulated AND NOT is_service;
    SELECT count(*) INTO serv_n FROM query_logs
    WHERE NOT is_simulated AND is_service;
    SELECT count(DISTINCT user_id_hash) INTO users_n FROM query_logs
    WHERE NOT is_simulated AND NOT is_service;

    RAISE NOTICE 'реальных запросов: %, служебных: %, разных сессий: %',
        real_n, serv_n, users_n;
END $$;