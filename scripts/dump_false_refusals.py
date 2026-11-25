"""Дамп признаков ложных отказов: что ретривер нашёл ДО гейта.

Зачем: refusal precision 0.947 достигнута каскадным гейтом по покрытию
токенов (0.35). Но гейт режет и верные ответы. Прежде чем поднимать или
опускать порог, нужно знать: ретривер НАШЁЛ золотой чанк, а гейт его
отсёк (тогда лечится порогом), или ретривер не нашёл вовсе (тогда порог
не поможет — нужен другой retrieval).

Для каждого in-scope вопроса прогоняем поиск ДВАЖДЫ:
  gate=0.0   — как если бы отсечки не было (видим результат ретривера);
  gate=COVERAGE_GATE — как в проде (видим фактическое поведение).

Пишет JSON и печатает таблицу. Ветвление по результату:
  - если большинство ложных отказов имеют gold в топ-3 при gate=0 →
    проблема в гейте, свип порога оправдан;
  - если нет → порог ни при чём, надо править retrieval.

Запуск:
    python scripts/dump_false_refusals.py --dsn "..." --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.search.encoder import Encoder  # noqa: E402
from app.search.search import COVERAGE_GATE, Searcher  # noqa: E402


def gold_hit(sources, gold: dict) -> bool:
    """Есть ли золотой чанк среди источников."""
    ids = set(gold.get("chunk_ids") or [])
    if not ids:
        return False
    return bool(ids & {s.chunk_id for s in sources})


def main() -> int:
    ap = argparse.ArgumentParser(description="Дамп ложных отказов")
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
    in_scope = [r for r in rows if not r["out_of_scope"] and not r.get("is_stale")]
    print(f"split={args.split}: in-scope={len(in_scope)}, "
          f"out-of-scope={sum(1 for r in rows if r['out_of_scope'])}, "
          f"stale={sum(1 for r in rows if r.get('is_stale'))}")

    enc = Encoder(args.model_dir)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    s = Searcher(args.dsn, enc, "dump-pepper", cfg)

    records = []
    for r in in_scope:
        gold = r.get("gold") or {}

        # 1) Как в проде: с гейтом.
        prod = s.search(r["query"], user_id="dump", use_cache=False)
        # 2) Без гейта: видим результат ретривера.
        open_ = s.search(r["query"], user_id="dump", use_cache=False, gate=0.0)

        f = open_.features
        rec = {
            "query": r["query"],
            "gold_section": gold.get("section_path"),
            "gold_chunk_ids": gold.get("chunk_ids") or [],
            "coverage": round(f.get("token_coverage", 0.0), 4),
            "cosine": round(f.get("max_sentence_sim", 0.0), 4),
            "margin": round(f.get("margin", 0.0), 4),
            "top1_chunk_sim": round(f.get("top1_chunk_sim", 0.0), 4),
            "rrf_top": round(f.get("rrf_top", 0.0), 4),
            "p_raw": round(open_.confidence, 4),
            "prod_type": prod.response_type,
            "prod_sources": [
                {"chunk_id": x.chunk_id, "section": x.section_path}
                for x in prod.sources[:3]
            ],
            "open_type": open_.response_type,
            "open_sources": [
                {"chunk_id": x.chunk_id, "section": x.section_path}
                for x in open_.sources[:3]
            ],
            # Ключевые флаги ветвления.
            "hit_prod": gold_hit(prod.sources, gold),
            "hit_open": gold_hit(open_.sources, gold),
            "gate_blocked": (
                prod.response_type == "refusal"
                and open_.response_type != "refusal"
            ),
        }
        records.append(rec)

    # --- Статистика для ветвления ---
    false_ref = [r for r in records if not r["hit_prod"]]
    blocked = [r for r in false_ref if r["gate_blocked"]]
    found_by_retriever = [r for r in false_ref if r["hit_open"]]
    lost_by_retriever = [r for r in false_ref if not r["hit_open"]]

    n_in = len(records)
    print(f"\n{'='*72}")
    print(f"ЛОЖНЫХ ОТКАЗОВ: {len(false_ref)} из {n_in}")
    print(f"  из них гейт отсёк верный ответ: {len(blocked)} из {n_in}")
    print(f"  ретривер нашёл золото (gate=0): {len(found_by_retriever)} из {n_in}")
    print(f"  ретривер НЕ нашёл золото:       {len(lost_by_retriever)} из {n_in}")

    print(f"\n{'cov':>5s} {'cos':>5s} {'mrg':>6s} {'p':>5s} {'гейт':>5s} "
          f"{'retr':>5s}  запрос / gold")
    print("-" * 96)
    for r in false_ref:
        print(
            f"{r['coverage']:5.2f} {r['cosine']:5.2f} {r['margin']:6.3f} "
            f"{r['p_raw']:5.2f} "
            f"{'ДА' if r['gate_blocked'] else '-':>5s} "
            f"{'ЕСТЬ' if r['hit_open'] else 'НЕТ':>5s}  "
            f"{r['query'][:44]} / {r['gold_section']}"
        )

    # --- Вывод для ветвления ---
    print(f"\n{'='*72}")
    if found_by_retriever:
        share = len(found_by_retriever) / len(false_ref) if false_ref else 0
        print(f"ВЕТВЛЕНИЕ: ретривер находит золото в {share:.0%} ложных отказов.")
        if share >= 0.5:
            print("  -> Гейт слишком строг. Свип порога по dev оправдан.")
        else:
            print("  -> Проблема в ретривере, а не в гейте. Свип порога")
            print("     даст мало: большинство золотых чанков не находятся.")
    else:
        print("ВЕТВЛЕНИЕ: ретривер не находит золото ни в одном отказе.")
        print("  -> Порог ни при чём. Нужен другой retrieval, не свип.")

    out = Path(args.out or f"reports/false_refusals_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": args.split,
        "n_in_scope": n_in,
        "n_false_refusals": len(false_ref),
        "n_gate_blocked": len(blocked),
        "n_retriever_found": len(found_by_retriever),
        "n_retriever_missed": len(lost_by_retriever),
        "coverage_gate": COVERAGE_GATE,
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nотчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
