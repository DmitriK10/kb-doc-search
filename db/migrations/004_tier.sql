-- ============================================================================
-- 004_tier.sql — флаг тира в журнале запросов
--
-- ЗАЧЕМ. Тир (success + low_confidence) — отдельный исход, но в query_logs
-- он неотличим от уверенного ответа: оба пишутся как response_type='success'.
-- Из-за этого нельзя посчитать СТРОГИЙ deflection (без тира) — а именно он
-- нужен для Time Saved, потому что цитата с плашкой на вопрос вне базы
-- не экономит время: сотрудник всё равно пойдёт к владельцу.
--
-- Строгий deflection = (success БЕЗ тира + stale) / total
-- Полный deflection = (success + stale) / total
--
-- Разница — это и есть цена тира, и она должна быть видна отдельной цифрой,
-- а не спрятана в общем success.
--
-- RAM: 0 (одна boolean-колонка). Стоимость: 0.
-- ============================================================================

ALTER TABLE query_logs
    ADD COLUMN IF NOT EXISTS low_confidence BOOLEAN NOT NULL DEFAULT FALSE;

COMMENT ON COLUMN query_logs.low_confidence IS
    'Тир: ответ выдан с пометкой «проверьте источник». Для строгого '
    'deflection такие ответы не считаются отвеченными: сотрудник всё '
    'равно пойдёт к владельцу.';

-- Партиальный индекс: строгий deflection считается по success без тира.
CREATE INDEX IF NOT EXISTS query_logs_confident
    ON query_logs (created_at DESC)
    WHERE response_type = 'success' AND NOT low_confidence;

-- Представление с обеими формулами deflection: одна точка правды,
-- чтобы панель и тексты не считали по-разному.
CREATE OR REPLACE VIEW v_deflection AS
SELECT
    count(*) AS total,
    count(*) FILTER (WHERE response_type = 'success'
                       AND NOT low_confidence) AS confident_success,
    count(*) FILTER (WHERE response_type = 'success'
                       AND low_confidence) AS tier,
    count(*) FILTER (WHERE response_type = 'stale') AS stale,
    count(*) FILTER (WHERE response_type = 'refusal') AS refusal,
    -- Строгий: только уверенные ответы и stale.
    round(
        (count(*) FILTER (WHERE response_type = 'success'
                            AND NOT low_confidence)
         + count(*) FILTER (WHERE response_type = 'stale'))::numeric
        / NULLIF(count(*), 0) * 100, 1
    ) AS strict_deflection_pct,
    -- Полный: плюс тир.
    round(
        (count(*) FILTER (WHERE response_type IN ('success', 'stale')))::numeric
        / NULLIF(count(*), 0) * 100, 1
    ) AS full_deflection_pct
FROM query_logs
WHERE is_simulated = FALSE;

COMMENT ON VIEW v_deflection IS
    'Обе формулы deflection по РЕАЛЬНЫМ запросам. Синтетика исключена: '
    'KPI считается только по is_simulated = FALSE.';

-- То же для демонстрационной истории — отдельным представлением,
-- чтобы её нельзя было случайно смешать с реальными.
CREATE OR REPLACE VIEW v_deflection_demo AS
SELECT
    count(*) AS total,
    count(*) FILTER (WHERE response_type = 'success'
                       AND NOT low_confidence) AS confident_success,
    count(*) FILTER (WHERE response_type = 'success'
                       AND low_confidence) AS tier,
    count(*) FILTER (WHERE response_type = 'stale') AS stale,
    count(*) FILTER (WHERE response_type = 'refusal') AS refusal,
    round(
        (count(*) FILTER (WHERE response_type = 'success'
                            AND NOT low_confidence)
         + count(*) FILTER (WHERE response_type = 'stale'))::numeric
        / NULLIF(count(*), 0) * 100, 1
    ) AS strict_deflection_pct,
    round(
        (count(*) FILTER (WHERE response_type IN ('success', 'stale')))::numeric
        / NULLIF(count(*), 0) * 100, 1
    ) AS full_deflection_pct
FROM query_logs
WHERE is_simulated = TRUE;

COMMENT ON VIEW v_deflection_demo IS
    'Те же формулы по демо-истории. Формулировки и даты синтетические, '
    'эмбеддинги и решения реальные.';