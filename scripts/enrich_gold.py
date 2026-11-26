"""Обогащение валидационного набора реальными chunk_id из БД.

Зачем это нужно и почему нельзя обойтись section_path:

Чанкер склеивает короткие пункты в пределах раздела. Пункты 3.1–3.4
регламента договоров становятся ОДНИМ чанком с section_path = '3.1'.
Если в разметке указать gold.section_path = '3.2', проверка «попал ли
правильный пункт» даст ложный промах: чанк существует, текст в нём есть,
но путь у него другой.

Поэтому разметку обогащаем реальными chunk_id: для каждого gold ищем
чанк, чей body содержит номер нужного пункта. Дальше eval сверяет
пересечение множеств chunk_id — это устойчиво к склейке и нарезке.

Запуск:
    python scripts/enrich_gold.py --dsn "..." --data-dir data
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]


def find_chunks(cur, doc_title: str, section_path: str) -> list[int]:
    """Ищет чанки, содержащие нужный пункт.

    Три попытки, от точной к широкой:
      1. section_path совпадает точно — чанк не склеен;
      2. body содержит «<section_path>.» — пункт внутри склеенного чанка;
      3. body содержит номер без точки — на случай «п. 4.2» в тексте.

    Возвращает список id (обычно один, иногда два при нарезке длинного
    пункта на части — тогда обе части валидны как источник).
    """
    # 1) точное совпадение пути
    cur.execute("""
        SELECT c.id FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.title = %s AND c.section_path = %s AND c.is_active
    """, (doc_title, section_path))
    ids = [r[0] for r in cur.fetchall()]
    if ids:
        return ids

    # 2) пункт внутри склеенного чанка
    pattern = re.escape(section_path) + r"\."
    cur.execute("""
        SELECT c.id FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.title = %s AND c.is_active AND c.body ~ %s
    """, (doc_title, r"(^|\n|\s)" + pattern))
    ids = [r[0] for r in cur.fetchall()]
    if ids:
        return ids

    # 3) свободное вхождение номера
    cur.execute("""
        SELECT c.id FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.title = %s AND c.is_active AND c.body LIKE %s
    """, (doc_title, f"%{section_path}%"))
    return [r[0] for r in cur.fetchall()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Обогащение gold реальными chunk_id")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    if psycopg is None:
        print("Нужен psycopg", file=sys.stderr)
        return 2

    data_dir = Path(args.data_dir)
    total, resolved, unresolved = 0, 0, []

    with psycopg.connect(args.dsn, autocommit=False) as conn:
        cur = conn.cursor()
        for name in ("validation_dev.jsonl", "validation_test.jsonl"):
            path = data_dir / name
            if not path.exists():
                continue
            rows = [json.loads(l) for l in
                    path.read_text(encoding="utf-8").splitlines() if l.strip()]
            out = []
            for r in rows:
                total += 1
                gold = r.get("gold")
                if gold:
                    # Междокументные вопросы (п. 7.5 брифа): золото — НЕСКОЛЬКО
                    # регламентов. Правильный ответ — цитата из любого из них,
                    # поэтому chunk_ids объединяются: иначе eval занизил бы
                    # Recall на таком золоте, посчитав верный ответ промахом.
                    if gold.get("multi"):
                        all_ids: list[int] = []
                        ok = True
                        for part in gold["multi"]:
                            ids = find_chunks(cur, part["doc_title"],
                                              part["section_path"])
                            if ids:
                                all_ids.extend(ids)
                            else:
                                ok = False
                                unresolved.append(
                                    f"{part['doc_title']} / {part['section_path']}")
                        if all_ids:
                            gold = dict(gold)
                            gold["chunk_ids"] = sorted(set(all_ids))
                            r["gold"] = gold
                            resolved += 1
                        continue
                    ids = find_chunks(cur, gold["doc_title"], gold["section_path"])
                    if ids:
                        gold = dict(gold)
                        gold["chunk_ids"] = ids
                        r["gold"] = gold
                        resolved += 1
                    else:
                        unresolved.append(
                            f"{gold['doc_title']} / {gold['section_path']}")
                out.append(r)
            path.write_text(
                "\n".join(json.dumps(x, ensure_ascii=False) for x in out),
                encoding="utf-8",
            )
        cur.close()

    print(f"записей: {total}, gold разрешён: {resolved}")
    if unresolved:
        print(f"НЕ разрешено ({len(unresolved)}):")
        for u in unresolved[:12]:
            print(f"  - {u}")
    return 0 if not unresolved else 1


if __name__ == "__main__":
    raise SystemExit(main())