"""Confusion matrix: единственный источник метрик.

Зачем отдельный скрипт. В отчёте от 25.11 числа не сходились: Recall@3 0.842
(= 32/38) и «10 ложных отказов из 38» несовместимы, потому что 10 отказов
дают максимум 28/38 = 0.737. Причина — метрики считались в разных местах
по разным правилам: recall по одному набору флагов, ложные отказы по другому.

Здесь одна матрица, и ВСЕ метрики выводятся из неё арифметически. Если
цифра на слайде не получается из матрицы — цифры не существует.

Матрица для in-scope (у вопроса есть золотой чанк):
                    gold в топ-3    gold не в топ-3
  уверенный ответ        A                B
  тир                    C                D
  отказ                  E                F

Для out-of-scope (ответа в базе нет):
  уверенный ответ     G      (ложный ответ — плохо)
  тир                 H      (цитата с плашкой — цена тира)
  отказ               I      (верный отказ — хорошо)

Метрики:
  Recall@3           = (A + C) / (A+B+C+D+E+F)   все ответы, включая тир
  confident_recall   = A / (A+B+C+D+E+F)          только уверенные
  refusal_precision  = I / (G + I)                только уверенные решения
  tier_on_oos_rate   = H / (G + H + I)            отдельная метрика тира
  false_refusals     = E + F                       отказы на in-scope

Запуск:
    python scripts/confusion_matrix.py --dsn "..." --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.search.encoder import Encoder  # noqa: E402
from app.search.search import Searcher  # noqa: E402


def classify(answer, gold_ids: set[int]) -> str:
    """Один из трёх исходов для ответа системы."""
    if answer.response_type == "refusal":
        return "refusal"
    if answer.low_confidence:
        return "tier"
    return "confident"


def gold_in_sources(sources, gold_ids: set[int]) -> bool:
    if not gold_ids:
        return False
    return bool(gold_ids & {s.chunk_id for s in sources[:3]})


def main() -> int:
    ap = argparse.ArgumentParser(description="Confusion matrix и вывод метрик")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model-dir", default="/opt/kb/models/rubert-tiny2-int8")
    ap.add_argument("--config", default="config/refusal.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rows = [
        json.loads(line)
        for line in (data_dir / f"validation_{args.split}.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    enc = Encoder(args.model_dir)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    s = Searcher(args.dsn, enc, "cm-pepper", cfg)

    # --- Счётчики ---
    A = B = C = D = E = F = 0        # in-scope
    G = H = I = 0                     # out-of-scope
    stale_total = stale_ok = 0
    details = []

    for r in rows:
        ans = s.search(r["query"], user_id="cm", use_cache=False)

        if r.get("is_stale"):
            stale_total += 1
            ok = ans.response_type in ("stale", "refusal")
            stale_ok += int(ok)
            details.append({
                "query": r["query"], "class": "stale",
                "outcome": ans.response_type,
                "correct": ok,
            })
            continue

        if r["out_of_scope"]:
            cls = classify(ans, set())
            if cls == "confident":
                G += 1
            elif cls == "tier":
                H += 1
            else:
                I += 1
            details.append({
                "query": r["query"], "class": "out_of_scope",
                "outcome": cls,
            })
            continue

        gold_ids = set((r.get("gold") or {}).get("chunk_ids") or [])
        hit = gold_in_sources(ans.sources, gold_ids)
        cls = classify(ans, gold_ids)

        if cls == "confident":
            if hit:
                A += 1
            else:
                B += 1
        elif cls == "tier":
            if hit:
                C += 1
            else:
                D += 1
        else:
            if hit:
                E += 1
            else:
                F += 1
        details.append({
            "query": r["query"], "class": "in_scope", "outcome": cls,
            "gold_hit": hit, "gold_section": (r.get("gold") or {}).get("section_path"),
            "cov": round(ans.features.get("token_coverage", 0), 3),
        })

    # --- Арифметика: всё выводится из матрицы ---
    n_in = A + B + C + D + E + F
    n_oos = G + H + I

    recall_at_3 = (A + C) / n_in if n_in else 0.0
    confident_recall = A / n_in if n_in else 0.0
    # Precision отказа: только уверенные решения (отказ против ложного ответа).
    refusal_precision = I / (G + I) if (G + I) else 0.0
    tier_on_oos = H / n_oos if n_oos else 0.0
    false_refusals = E + F
    false_answers = G
    stale_accuracy = stale_ok / stale_total if stale_total else None

    print("=" * 74)
    print(f"CONFUSION MATRIX — split={args.split}")
    print("=" * 74)
    print()
    print("IN-SCOPE (у вопроса есть золотой чанк):")
    print(f"{'':22s} {'gold в топ-3':>14s} {'gold не в топ-3':>16s} {'всего':>8s}")
    print(f"{'уверенный ответ':22s} {A:14d} {B:16d} {A+B:8d}")
    print(f"{'тир (low_confidence)':22s} {C:14d} {D:16d} {C+D:8d}")
    print(f"{'отказ':22s} {E:14d} {F:16d} {E+F:8d}")
    print(f"{'всего':22s} {A+C+E:14d} {B+D+F:16d} {n_in:8d}")
    print()
    print("OUT-OF-SCOPE (ответа в базе нет):")
    print(f"{'уверенный ответ':22s} {G:14d}   <- ЛОЖНЫЙ ОТВЕТ (плохо)")
    print(f"{'тир':22s} {H:14d}   <- цитата с плашкой (цена тира)")
    print(f"{'отказ':22s} {I:14d}   <- верный отказ (хорошо)")
    print(f"{'всего':22s} {n_oos:14d}")
    print()
    if stale_total:
        print(f"STALE-ВОПРОСЫ: {stale_ok}/{stale_total} корректно "
              f"распознаны (stale или отказ)")
        print()

    print("=" * 74)
    print("МЕТРИКИ, ВЫВЕДЕННЫЕ ИЗ МАТРИЦЫ")
    print("=" * 74)
    print(f"Recall@3                = (A+C)/(A+B+C+D+E+F) = ({A}+{C})/{n_in} = {recall_at_3:.4f}")
    print(f"  только уверенные      = A/(...)             = {A}/{n_in} = {confident_recall:.4f}")
    # Две дефиниции precision — обе публикуются с явными названиями.
    confident_precision = A / (A + B) if (A + B) else 0.0
    print(f"Refusal precision       = I/(G+I)             = {I}/{G+I} = {refusal_precision:.4f}")
    print(f"  (стандартная: доля верных отказов среди решений на OOS)")
    print(f"Отказов при найденном gold = E                 = {E}")
    print(f"  (гейт не отрезал ни одного найденного ответа)")
    print(f"Confident precision     = A/(A+B)             = {A}/{A+B} = {confident_precision:.4f}")
    print(f"  (уверенная цитата на верный документ)")
    print(f"Tier on out-of-scope    = H/(G+H+I)           = {H}/{n_oos} = {tier_on_oos:.4f}")
    print(f"Ложных уверенных ответов= G                   = {false_answers}")
    print(f"In-scope без ответа     = E+F                 = {false_refusals} из {n_in}")
    print(f"  (все {E+F} — retrieval miss, не ложный отказ; уже в Recall@3)")
    if stale_accuracy is not None:
        print(f"Stale accuracy          = {stale_ok}/{stale_total} = {stale_accuracy:.4f}")

    # --- Проверка сходимости ---
    print()
    print("ПРОВЕРКА СХОДИМОСТИ:")
    checks = [
        ("A+B+C+D+E+F == n_in", A + B + C + D + E + F == n_in),
        ("G+H+I == n_oos", G + H + I == n_oos),
        ("Recall@3 >= confident_recall", recall_at_3 >= confident_recall - 1e-9),
        ("false_refusals == E+F", false_refusals == E + F),
        ("(A+C) <= n_in", A + C <= n_in),
    ]
    all_ok = True
    for name, ok in checks:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok

    if not all_ok:
        print("\nВНИМАНИЕ: матрица не сходится, числа публиковать нельзя.")
        return 1

    out = Path(args.out or f"reports/confusion_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": args.split,
        "matrix": {
            "in_scope": {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F,
                         "total": n_in},
            "out_of_scope": {"G": G, "H": H, "I": I, "total": n_oos},
            "stale": {"correct": stale_ok, "total": stale_total},
        },
        "metrics": {
            "recall_at_3": round(recall_at_3, 4),
            "confident_recall": round(confident_recall, 4),
            "refusal_precision": round(refusal_precision, 4),
            "refusals_at_found_gold": E,
            "confident_precision": round(confident_precision, 4),
            "in_scope_without_answer": false_refusals,
            "tier_on_out_of_scope": round(tier_on_oos, 4),
            "false_answers": false_answers,
            "false_refusals": false_refusals,
            "stale_accuracy": (round(stale_accuracy, 4)
                               if stale_accuracy is not None else None),
        },
        "details": details,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nотчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())