"""Проверка RRF, реранка и признаков уверенности на синтетике.

Запуск: python tests/test_fusion.py
Печатает JSON-отчёт; код возврата 1 при провале.

Зачем синтетика: pgvector на этой машине не установлен (нужны права
администратора), но логика слияния от него не зависит — она работает
с числами. Проверяем ровно то, что может быть проверено без БД:
  - RRF даёт более высокий score документу, найденному ОБЕИМИ ветками;
  - k=60 влияет монотонно;
  - реранк выбирает максимум по предложениям, а не среднее;
  - margin различает «уверенный ответ» и «шум»;
  - sigmoid устойчив на краях.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.search.fusion import (  # noqa: E402
    RRF_K,
    confidence,
    confidence_features,
    pick_best_sentence,
    rrf_fuse,
    sentence_rerank,
    sigmoid,
)

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail and ok else ""))
    if not ok:
        FAILS.append(name)


def main() -> int:
    print("-- RRF --")
    # Документ 7 найден обеими ветками (ранг 1 в FTS, ранг 3 в векторе).
    # Документ 1 — только FTS ранг 1. Совпадение в двух ветках должно победить.
    fts = [7, 1, 2, 3]
    vec = [5, 6, 7, 8]
    fused = dict(rrf_fuse(fts, vec))

    check("документ из обеих веток получает вклад от обеих",
          abs(fused[7] - (1 / (RRF_K + 1) + 1 / (RRF_K + 3))) < 1e-12,
          f"score(7)={fused[7]:.6f}")
    check("документ только из FTS ранга 1 ниже, чем документ 7 (обе ветки)",
          fused[1] < fused[7],
          f"score(1)={fused[1]:.6f} < score(7)={fused[7]:.6f}")
    check("документ только из вектора ранга 1 ниже, чем документ 7",
          fused[5] < fused[7],
          f"score(5)={fused[5]:.6f}")

    order = [cid for cid, _ in rrf_fuse(fts, vec)]
    check("первый в выдаче — документ из обеих веток", order[0] == 7, f"order={order}")

    # Монотонность по k: больше k — меньше разрыв между рангами.
    s60 = dict(rrf_fuse(fts, vec, k=60))
    s10 = dict(rrf_fuse(fts, vec, k=10))
    ratio60 = s60[7] / s60[1]
    ratio10 = s10[7] / s10[1]
    check("с ростом k разрыв между рангами сглаживается",
          ratio60 < ratio10, f"k=60: {ratio60:.3f} < k=10: {ratio10:.3f}")

    # Пустые ветки: RRF не должен падать.
    check("пустые списки -> пустой результат", rrf_fuse([], []) == [])
    check("одна ветка пуста -> работает вторая",
          [cid for cid, _ in rrf_fuse([], [3, 4])] == [3, 4])

    print("\n-- sentence-level реранк --")
    rng = np.random.default_rng(42)
    dim = 312
    q = rng.normal(size=dim).astype(np.float32)
    q /= np.linalg.norm(q)

    # Чанк 10: одно предложение почти совпадает с запросом, остальные — шум.
    # Чанк 20: все предложения «средне похожи». Средний cosine у 20 выше,
    # но max — у 10. Реранк обязан выбрать 10: нам нужен один точный пункт,
    # а не «в среднем про то же».
    # ВАЖНО про масштаб шума: на 312 измерениях шум с sigma=0.05 имеет норму
    # 0.05*sqrt(312) ≈ 0.88, то есть сопоставим с самим вектором (норма 1).
    # Тогда cosine падает до ~0.75, и тест проверяет не то, что задумано.
    # Чтобы получить cosine ≈ 0.99, нужен шум с нормой ~0.14, то есть
    # sigma ≈ 0.14/sqrt(312) ≈ 0.008.
    near = q + rng.normal(scale=0.008, size=dim).astype(np.float32)
    near /= np.linalg.norm(near)
    mid = q * 0.55 + rng.normal(scale=0.8, size=dim).astype(np.float32)
    mid /= np.linalg.norm(mid)

    sentences = [
        (10, 0, rng.normal(size=dim)),
        (10, 1, near),
        (10, 2, rng.normal(size=dim)),
        (20, 0, mid), (20, 1, mid), (20, 2, mid), (20, 3, mid),
    ]
    scores = sentence_rerank(q, sentences)

    check("реранк берёт максимум по предложениям, а не среднее",
          scores[10] > scores[20],
          f"max(10)={scores[10]:.3f} > max(20)={scores[20]:.3f}")
    check("max cosine для почти совпадающего предложения близок к 1",
          scores[10] > 0.95, f"{scores[10]:.3f}")

    best = pick_best_sentence(q, [(10, "точный пункт", near),
                                  (20, "общий раздел", mid)])
    check("pick_best_sentence возвращает текст победителя",
          best is not None and best[2] == "точный пункт", str(best)[:70])

    check("пустой список предложений -> пустой результат",
          sentence_rerank(q, []) == {})
    check("нулевой вектор запроса не даёт деления на ноль",
          sentence_rerank(np.zeros(dim, dtype=np.float32), sentences) == {})

    print("\n-- признаки уверенности --")
    # Уверенный ответ: высокий max, большой margin, согласие веток.
    f_conf = confidence_features(
        rerank={1: 0.82, 2: 0.41},
        chunk_sims={1: 0.70, 2: 0.38},
        rrf_scores=[(1, 2 / 61), (2, 1 / 61)],
        fts_exact=True,
    )
    # Шум: два почти равных кандидата, низкий max.
    f_noise = confidence_features(
        rerank={3: 0.58, 4: 0.56},
        chunk_sims={3: 0.50, 4: 0.49},
        rrf_scores=[(3, 1 / 61), (4, 1 / 61)],
        fts_exact=False,
    )

    check("margin у уверенного ответа большой",
          f_conf["margin"] > 0.3, f"margin={f_conf['margin']:.2f}")
    check("margin у шума малый", f_noise["margin"] < 0.05,
          f"margin={f_noise['margin']:.3f}")
    check("margin — это разрыв top1-top2, а не абсолютная величина",
          abs(f_conf["margin"] - (0.82 - 0.41)) < 1e-9)
    check("fts_exact кодируется как 1.0/0.0",
          f_conf["fts_exact"] == 1.0 and f_noise["fts_exact"] == 0.0)
    check("rrf_top нормирован в 0..1",
          0.0 <= f_conf["rrf_top"] <= 1.0 and 0.0 <= f_noise["rrf_top"] <= 1.0,
          f"conf={f_conf['rrf_top']:.2f}, noise={f_noise['rrf_top']:.2f}")

    print("\n-- логистика --")
    # Веса-заглушки: проверяем механику, не качество (его даст калибратор).
    w = {"b0": -6.0, "max_sentence_sim": 8.0, "margin": 2.0,
         "top1_chunk_sim": 2.0, "rrf_top": 1.0, "fts_exact": 0.3}
    p_conf = confidence(f_conf, w)
    p_noise = confidence(f_noise, w)
    check("уверенный ответ получает вероятность выше шума",
          p_conf > p_noise, f"{p_conf:.3f} > {p_noise:.3f}")
    check("вероятность в диапазоне 0..1",
          0.0 <= p_conf <= 1.0 and 0.0 <= p_noise <= 1.0)

    check("sigmoid(0) == 0.5", abs(sigmoid(0.0) - 0.5) < 1e-12)
    check("sigmoid не переполняется на больших |z|",
          0.0 < sigmoid(1000.0) <= 1.0 and 0.0 <= sigmoid(-1000.0) < 1.0,
          f"sig(1000)={sigmoid(1000.0):.6f}, sig(-1000)={sigmoid(-1000.0):.6f}")
    check("sigmoid монотонен", sigmoid(-2) < sigmoid(0) < sigmoid(2))

    print("\n" + "=" * 58)
    if FAILS:
        print(f"ПРОВАЛЕНО {len(FAILS)}: " + "; ".join(FAILS))
        return 1
    print("Все проверки пройдены")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())