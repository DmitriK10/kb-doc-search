"""Слияние результатов поиска: RRF и sentence-level реранк.

Обе функции — чистый numpy: они не ходят в БД и не вызывают модель. Это
позволяет проверять их на синтетических векторах (tests/test_fusion.py),
не поднимая pgvector — что важно, потому что на дев-машине pgvector не
устанавливается без прав администратора.

Потребление RAM:
  RRF: O(k), k = 50+50 кандидатов -> единицы КБ.
  Реранк: матрица (n_sentences × 312) float32. 200 предложений = 250 КБ.
Скорость на 2 vCPU:
  RRF: ~0.05 мс.
  Реранк 200 предложений: 8–15 мс (один matmul).
"""

from __future__ import annotations

import numpy as np

from app.utils.vec import normalize_rows, to_f32

# k=60 — классическое значение RRF (Cormack et al., 2009), зафиксировано в ТЗ.
# Меньше k — резче доминирует топ-1; больше k — ровнее. Менять только по A/B.
RRF_K = 60


def rrf_fuse(
    fts_ids: list[int],
    vec_ids: list[int],
    k: int = RRF_K,
    weights: tuple[float, float] = (1.0, 1.0),
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion двух ранжированных списков.

    score(d) = Σ w_i / (k + rank_i(d)), rank с 1.

    Смысл конструкции: документ, найденный ОБЕИМИ ветками, всегда получает
    больше, чем документ с идеальным первым местом только в одной. Это и есть
    механизм согласия — ровно то, что защищает от того, чтобы FTS-шум или
    векторный шум в одиночку вытащил нерелевантный чанк наверх.

    Возвращает [(id, score)] по убыванию score.
    """
    w_fts, w_vec = weights
    scores: dict[int, float] = {}

    for rank, cid in enumerate(fts_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + w_fts / (k + rank)
    for rank, cid in enumerate(vec_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + w_vec / (k + rank)

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def sentence_rerank(
    query_vec,
    sentences: list[tuple[int, int, object]],
) -> dict[int, float]:
    """Максимальный косинус между запросом и предложениями каждого чанка.

    sentences: [(chunk_id, seq, embedding), ...] — предвычисленные на
    индексации векторы предложений. Модель на запросе НЕ вызывается:
    это главная экономия латентности (−150 мс) и RAM (−200 МБ).

    Почему максимум, а не среднее: нам нужен один точный пункт регламента,
    а не «в среднем про то же». Чанк с одним идеальным предложением и шумом
    вокруг полезнее чанка, где всё «средне похоже».

    to_f32 обязателен: halfvec из psycopg приходит объектом Halfvec,
    np.asarray на нём падает или даёт мусор (правка К5).
    """
    if not sentences:
        return {}

    q = to_f32(query_vec).astype(np.float32)
    qn = float(np.linalg.norm(q))
    if qn < 1e-9:
        return {}
    q = q / qn

    mat = normalize_rows(np.vstack([to_f32(s[2]) for s in sentences]))
    sims = mat @ q

    best: dict[int, float] = {}
    for (chunk_id, _seq, _emb), sim in zip(sentences, sims):
        val = float(sim)
        if val > best.get(chunk_id, -2.0):
            best[chunk_id] = val
    return best


def pick_best_sentence(
    query_vec,
    sentences: list[tuple[int, str, object]],
) -> tuple[int, int, str, float] | None:
    """Лучшее предложение среди всех чанков: (chunk_id, idx, text, sim).

    Нужна для показа цитаты: пользователю отдаём предложение-победитель,
    а не чанк целиком, если чанк длинный (правка B5 — «цитируем пункт,
    а не обрезок»).
    """
    if not sentences:
        return None

    q = to_f32(query_vec).astype(np.float32)
    qn = float(np.linalg.norm(q))
    if qn < 1e-9:
        return None
    q = q / qn

    mat = normalize_rows(np.vstack([to_f32(s[2]) for s in sentences]))
    sims = mat @ q

    idx = int(np.argmax(sims))
    chunk_id, text, _emb = sentences[idx]
    return chunk_id, idx, text, float(sims[idx])


def final_rank(
    rerank: dict[int, float],
    coverage: dict[int, float],
    candidate_ids: list[int],
    coverage_weight: float = 0.12,
    number_match: dict[int, float] | None = None,
    number_weight: float = 0.18,
) -> list[tuple[int, float]]:
    """Итоговый порядок источников: смысловая близость + покрытие терминов.

    Зачем не только max_sentence_sim. Реальный провал, найденный на стенде:
    на вопрос «кто подписывает агентский договор свыше миллиона» в базе есть
    два соседних пункта матрицы согласования:
        п.5#т1.3 «Агентский договор, свыше 1 000 000 руб. → Финансовый директор»
        п.5#т1.4 «Любой договор, свыше 5 000 000 руб. → Генеральный директор»
    Оба содержат «договор» и «свыше», поэтому косинусы почти равны
    (0.614 против 0.615), и RRF ставит наверх НЕ ТОТ пункт. Разница в
    доли тысячной, то есть шум.

    Различает их покрытие токенов: у правильного пункта есть слово
    «агентский», у неправильного — нет (0.60 против 0.40). Поэтому
    покрытие входит в итоговую оценку с весом, который перевешивает
    шумовой разрыв косинусов, но не перебивает смысл.

    coverage_weight=0.12: при разнице покрытия 0.20 это даёт +0.024,
    что уверенно больше типичного шумового разрыва косинусов (0.005-0.01),
    но меньше разницы между «похоже» и «не похоже» (0.15+).

    number_match — доля чисел запроса, найденных в чанке (0..1). Нужен,
    потому что косинус НЕ РАЗЛИЧАЕТ СУММЫ: «до 100 000 руб.» и «свыше
    1 000 000 руб.» — почти один и тот же текст для энкодера. На стенде
    вопрос «агентский договор свыше миллиона» выдавал пункт про 5 000 000.
    Числовое совпадение решает это детерминированно.
    Вес 0.18: разница покрытия даёт максимум 0.12, а суммы —
    единственное, что отличает пункты матрицы друг от друга,
    поэтому число весит больше покрытия.

    Возвращает [(chunk_id, score)] по убыванию.
    """
    nm = number_match or {}
    scored = []
    for cid in candidate_ids:
        sim = rerank.get(cid, 0.0)
        cov = coverage.get(cid, 1.0)
        num = nm.get(cid, 1.0)
        scored.append((cid, sim + coverage_weight * cov + number_weight * num))
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored


def confidence_features(
    rerank: dict[int, float],
    chunk_sims: dict[int, float],
    rrf_scores: list[tuple[int, float]],
    fts_exact: bool,
    token_coverage: float = 1.0,
) -> dict[str, float]:
    """Признаки для калиброванного порога отказа.

    Почему не один скаляр: одиночный max-cosine на rubert-tiny2 даёт
    30–40% ложных отказов — «похожий, но не тот» пункт и «точный» пункт
    попадают в один диапазон. Признаки разделяют эти случаи.

      max_sentence_sim — главный: максимум по предложениям top-1;
      top1_chunk_sim   — косинус запроса к чанку целиком, грубее, но устойчив;
      margin           — разрыв top1 − top2. Если 0.58 против 0.56, это шум,
                         а не ответ, и margin показывает это там, где
                         абсолютный порог молчит;
      rrf_top          — нормированный RRF (согласие двух веток поиска);
      fts_exact        — 1.0, если сработало точное полнотекстовое совпадение;
      token_coverage   — доля содержательных токенов запроса, найденных
                         в тексте чанка (по леммам).

    Про token_coverage — это ответ на реальную проблему, найденную на
    валидации. Запрос «как оформить командировку за границу» семантически
    почти неотличим от «как оформить командировку»: документ про поездки
    по России даёт высокий косинус, и система отвечает по нему, хотя про
    заграницу там нет ни слова. Косинус этого не видит, а покрытие токенов
    видит: «границу» в чанке отсутствует, coverage падает.

    Ключи обязаны совпадать с FEATURES в scripts/eval.py и scripts/calibrate.
    """
    ordered = [cid for cid, _ in rrf_scores]
    top1 = ordered[0] if ordered else None
    top2 = ordered[1] if len(ordered) > 1 else None

    sim1 = rerank.get(top1, 0.0) if top1 is not None else 0.0
    sim2 = rerank.get(top2, 0.0) if top2 is not None else 0.0

    rrf_max = rrf_scores[0][1] if rrf_scores else 0.0
    # Потолок при двух ветках: 2/(k+1). Нормируем в 0..1.
    rrf_ceiling = 2.0 / (RRF_K + 1)
    rrf_norm = rrf_max / rrf_ceiling if rrf_ceiling > 0 else 0.0

    return {
        "max_sentence_sim": sim1,
        "top1_chunk_sim": chunk_sims.get(top1, 0.0) if top1 is not None else 0.0,
        "margin": sim1 - sim2,
        "rrf_top": min(rrf_norm, 1.0),
        "fts_exact": 1.0 if fts_exact else 0.0,
        "token_coverage": max(0.0, min(1.0, token_coverage)),
    }


def sigmoid(z: float) -> float:
    """Логистическая функция, устойчивая на больших |z|.

    Наивное 1/(1+exp(-z)) переполняется при z < -709. Здесь ветвление
    по знаку: exp берётся только от неположительного аргумента.
    """
    if z >= 0:
        return float(1.0 / (1.0 + np.exp(-min(z, 700.0))))
    ez = np.exp(max(z, -700.0))
    return float(ez / (1.0 + ez))


def confidence(features: dict[str, float], weights: dict[str, float]) -> float:
    """Калиброванная вероятность «ответ надёжен».

    p = sigmoid(b0 + Σ w_i·f_i). Веса — из config/refusal.json,
    получены scripts/calibrate.py на валидационном наборе.
    """
    z = weights.get("b0", 0.0)
    for name, value in features.items():
        z += weights.get(name, 0.0) * value
    return sigmoid(z)


def is_confident(
    features: dict[str, float],
    weights: dict[str, float],
    threshold: float,
) -> tuple[bool, float]:
    """Решение «отвечаем или отказываем» + вероятность.

    Возвращает (answer, p). Отказ — не ошибка, а штатный исход:
    выдуманный ответ дороже отказа (см. бизнес-контекст).
    """
    p = confidence(features, weights)
    return p >= threshold, p