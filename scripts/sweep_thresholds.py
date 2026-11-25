"""Свип порогов каскада по dev с целевой precision >= 0.90.

Правило выбора (зафиксировано до запуска, чтобы не подгонять результат):
  1. Считаем только на dev. Test не трогаем — он single-shot.
  2. Среди точек, где refusal precision >= 0.90, берём максимальный Recall@3.
  3. Если таких точек нет — берём точку с максимальным произведением
     recall × precision и печатаем предупреждение.
  4. Порог логистики и gate по покрытию подбираются СОВМЕСТНО: они
     влияют друг на друга, раздельный подбор даёт ложный оптимум.

Запуск:
    python scripts/sweep_thresholds.py --dsn "..." --split dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.search.encoder import Encoder  # noqa: E402
from app.search.fusion import confidence  # noqa: E402
from app.search.search import Searcher  # noqa: E402

TARGET_PRECISION = 0.90


def collect(searcher: Searcher, records: list[dict]) -> list[dict]:
    """Прогон набора БЕЗ гейта и без порога: собираем сырые признаки.

    Гейт и порог применяются потом в памяти, иначе пришлось бы гонять
    поиск на каждую комбинацию порогов (сотни прогонов вместо одного).
    """
    out = []
    for rec in records:
        ans = searcher.search(
            rec["query"], user_id="sweep", use_cache=False, gate=0.0
        )
        f = ans.features
        out.append({
            "out_of_scope": rec["out_of_scope"],
            "is_stale": rec.get("is_stale", False),
            "hit": bool(
                set(rec.get("gold", {}).get("chunk_ids") or [])
                & {s.chunk_id for s in ans.sources[:3]}
            ) if rec.get("gold") else False,
            "features": f,
            "p": ans.confidence,
        })
    return out


def evaluate(rows: list[dict], gate: float, thr: float, weights: dict) -> dict:
    """Метрики при заданных gate и пороге."""
    tp = fp = tn = fn = 0
    for r in rows:
        if r["is_stale"]:
            continue
        # Порог логистики — отдельно от gate: если веса пересчитаны,
        # p пересчитывается, иначе свип меряет не то.
        p = confidence(r["features"], weights) if weights else r["p"]
        answered = p >= thr and r["features"].get("token_coverage", 1.0) >= gate
        if r["out_of_scope"]:
            if answered:
                fp += 1
            else:
                tn += 1
        else:
            if answered:
                if r["hit"]:
                    tp += 1
                else:
                    fn += 1
            else:
                fn += 1
    n_in = tp + fn
    n_out = fp + tn
    return {
        "gate": round(gate, 3),
        "threshold": round(thr, 3),
        "recall@3": round(tp / n_in, 4) if n_in else 0.0,
        "refusal_precision": round(tn / n_out, 4) if n_out else 0.0,
        "false_answers": fp,
        "false_refusals": fn,
        "n_in_scope": n_in,
        "n_out_scope": n_out,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Свип порогов по dev")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model-dir", default="/opt/kb/models/rubert-tiny2-int8")
    ap.add_argument("--config", default="config/refusal.json")
    ap.add_argument("--out", default="reports/sweep_dev.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rows_in = [
        json.loads(line)
        for line in (data_dir / f"validation_{args.split}.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"split={args.split}: {len(rows_in)} записей")

    enc = Encoder(args.model_dir)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    weights = cfg.get("weights", {})
    s = Searcher(args.dsn, enc, "sweep-pepper", cfg)

    print("прогон (один раз, без гейта)...")
    rows = collect(s, rows_in)

    # --- Сетка ---
    gates = [round(g, 2) for g in np.arange(0.15, 0.65, 0.05)]
    thrs = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]

    results = []
    for gate in gates:
        for thr in thrs:
            results.append(evaluate(rows, gate, thr, weights))

    # --- Выбор по правилу ---
    good = [r for r in results if r["refusal_precision"] >= TARGET_PRECISION]
    if good:
        # Максимальный Recall; при равенстве — меньший gate (мягче отсечка).
        good.sort(key=lambda r: (-r["recall@3"], r["gate"], r["threshold"]))
        best = good[0]
        reason = f"precision >= {TARGET_PRECISION} достигнута"
    else:
        results.sort(
            key=lambda r: -(r["recall@3"] * r["refusal_precision"])
        )
        best = results[0]
        reason = (f"НЕТ точки с precision >= {TARGET_PRECISION}; "
                  f"взят максимум recall × precision")

    print(f"\n{'gate':>5s} {'thr':>5s} {'recall':>7s} {'prec':>6s} "
          f"{'FP':>4s} {'FN':>4s}  комментарий")
    print("-" * 72)
    # Показываем лучшие 15 по recall среди проходящих precision
    shown = sorted(good or results, key=lambda r: -r["recall@3"])[:15]
    for r in shown:
        mark = " <== ВЫБРАН" if (r["gate"] == best["gate"]
                                and r["threshold"] == best["threshold"]) else ""
        print(f"{r['gate']:5.2f} {r['threshold']:5.2f} "
              f"{r['recall@3']:7.3f} {r['refusal_precision']:6.3f} "
              f"{r['false_answers']:4d} {r['false_refusals']:4d}{mark}")

    print(f"\n{'='*72}")
    print(f"ВЫБРАН: gate={best['gate']:.2f}, threshold={best['threshold']:.2f}")
    print(f"  {reason}")
    print(f"  Recall@3          {best['recall@3']:.3f}")
    print(f"  Refusal precision {best['refusal_precision']:.3f}")
    print(f"  Ложных ответов    {best['false_answers']} из {best['n_out_scope']}")
    print(f"  Ложных отказов    {best['false_refusals']} из {best['n_in_scope']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": args.split,
        "target_precision": TARGET_PRECISION,
        "chosen": best,
        "reason": reason,
        "grid": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nотчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())