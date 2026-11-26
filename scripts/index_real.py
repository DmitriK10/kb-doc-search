"""Индексация РЕАЛЬНЫХ регламентов хакатона.

Отличие от синтетики:
  - нет стилей Title/Heading (все абзацы Normal), нет шапки с версией;
  - нумерация пунктов есть, но заголовки разделов не размечены;
  - таблицы «Обязательно / При наличии» нужно склеивать в ОДИН чанк
    (иначе половина смысла теряется: «обязательно» без «при наличии»);
  - владельцы заданы не в документе, а в брифе — подставляем из конфига.

Версий у реальных регламентов нет, поэтому valid_from = дата индексации,
valid_to = NULL, а version = «актуальная». Это честно: у документа нет
истории версий, и придумывать её нельзя.

Запуск:
    python scripts/index_real.py --src /path/to/materials --dsn "..."
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.search.encoder import Encoder, to_pgvector  # noqa: E402

try:
    from docx import Document
except ImportError:  # pragma: no cover
    Document = None  # type: ignore[assignment]


# Владельцы по документам — из брифа, с указанием пункта-основания.
# Бриф прямо называет ответственных для каждого регламента.
OWNERS: dict[str, dict[str, str]] = {
    "Пресейловая деятельность Дирекции ИБ": {
        "owner_unit": "Дирекция ИБ",
        "owner_role": "Пресейл-архитектор (ПА)",
        "owner_contact": "quality@company.ru",
        "basis": "п. 4.1–4.2: согласование ТКП с ПА; руководитель департамента — только п. 4.2.2",
    },
    "Проектная деятельность Дирекции ИБ": {
        "owner_unit": "Дирекция ИБ",
        "owner_role": "Координатор департамента",
        "owner_contact": "quality@company.ru",
        "basis": "п. 2.1: назначение РП выполняет Координатор департамента",
    },
    "Проектная деятельность ДБРЦТ": {
        "owner_unit": "Департамент бизнес-решений и цифровой трансформации",
        "owner_role": "Ресурс-менеджер / Архитектор",
        "owner_contact": "quality@company.ru",
        "basis": "п. 2.1: первичную валидацию проводит РМ/Архитектор",
    },
    "Организация проектной деятельности Тех дирекции": {
        "owner_unit": "Техническая дирекция",
        "owner_role": "РМ Департамента технологической поддержки продаж",
        "owner_contact": "quality@company.ru",
        "basis": "п. 2.1: заявка назначается на Ресурсного менеджера ДТПП",
    },
}

# Заголовок документа — первый непустой абзац, если он не пункт.
_ITEM = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+(.*)$")
_HEADING_NUM = re.compile(r"^(\d+)\.\s+([А-ЯЁA-Z].*)$")


def _is_heading(text: str) -> bool:
    """Похоже ли на заголовок раздела.

    В реальных регламентах стилей нет, поэтому заголовок определяем по
    форме: «1. Название» без точки в конце, до 80 символов, начинается
    с заглавной буквы. Пункты нумеруются «1.1.», заголовки — «1.».
    """
    m = _HEADING_NUM.match(text)
    if not m:
        return False
    title = m.group(2).strip()
    if len(text) > 90 or title.endswith("."):
        return False
    # Заголовок не содержит глаголов-сказуемых в начале предложения
    # и не заканчивается точкой с запятой (признак перечисления).
    return len(title) > 3


def parse_real_docx(path: Path) -> tuple[str, list[dict]]:
    """Разбирает реальный регламент в список структурных блоков.

    Возвращает (title, blocks), где block = {kind, text, section_path,
    section_title, table_id}.
    """
    doc = Document(str(path))
    paras = []
    for p in doc.paragraphs:
        t = " ".join(p.text.split())
        if t:
            paras.append(t)

    title = paras[0] if paras else path.stem
    blocks: list[dict] = []
    h1_path = ""
    h1_title = ""
    h2_path = ""
    h2_title = ""

    for t in paras[1:]:
        if _is_heading(t):
            m = _HEADING_NUM.match(t)
            h1_path = m.group(1)
            h1_title = m.group(2).strip()
            h2_path, h2_title = "", ""
            blocks.append({
                "kind": "h1", "text": t,
                "section_path": h1_path, "section_title": h1_title,
            })
            continue

        m = _ITEM.match(t)
        if m and len(m.group(1)) <= 12:
            num, body = m.group(1), m.group(2).strip()
            if not body:
                continue
            # Подпункт N.M.K — уточняем уровень, но путь оставляем как есть.
            blocks.append({
                "kind": "item", "text": body,
                "section_path": num,
                "section_title": h2_title or h1_title or title,
            })
            continue

        # Ненумерованный абзац: продолжение предыдущего пункта или пояснение.
        # Приклеиваем к предыдущему пункту, а не создаём отдельный чанк:
        # в реальных регламентах так оформлены перечисления внутри пункта.
        if blocks and blocks[-1]["kind"] == "item":
            blocks[-1]["text"] = blocks[-1]["text"] + " " + t
        else:
            blocks.append({
                "kind": "item", "text": t,
                "section_path": h1_path or "1",
                "section_title": h1_title or title,
            })

    # --- Таблицы: склеиваем в ОДИН чанк на таблицу ---
    # Бриф: таблицы «Обязательно / При наличии» — это одна мысль,
    # разбивать её построчно нельзя: «обязательно» без «при наличии»
    # теряет смысл.
    for ti, table in enumerate(doc.tables, 1):
        rows = []
        for row in table.rows:
            cells = [" ".join(c.text.split()) for c in row.cells]
            if any(cells):
                rows.append(cells)
        if not rows:
            continue
        header = rows[0]
        lines = [" | ".join(header)]
        for row in rows[1:]:
            pairs = [f"{h}: {v}" for h, v in zip(header, row) if v]
            if pairs:
                lines.append("; ".join(pairs))
        blocks.append({
            "kind": "table",
            "text": "\n".join(lines),
            "section_path": f"{h1_path}#т{ti}" if h1_path else f"т{ti}",
            "section_title": h1_title or title,
        })

    return title, blocks


def chunk_blocks(title: str, blocks: list[dict],
                 max_chars: int = 1200) -> list[dict]:
    """Группирует блоки в чанки по разделу, склеивая короткие пункты.

    Лимит 1200 символов выше синтетического (1000), потому что реальные
    пункты длиннее и дробление теряет контекст. Пункты одного раздела
    склеиваются, пока не исчерпан лимит.
    """
    chunks: list[dict] = []
    buf: list[str] = []
    buf_len = 0
    cur_section = ""
    cur_title = ""

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        body = "\n".join(buf).strip()
        if body:
            chunks.append({
                "body": body,
                "section_path": cur_section,
                "section_title": cur_title,
                "char_len": len(body),
            })
        buf = []
        buf_len = 0

    for b in blocks:
        if b["kind"] == "h1":
            flush()
            cur_section = b["section_path"]
            cur_title = b["section_title"]
            continue

        text = b["text"].strip()
        if not text:
            continue
        # Смена раздела — новый чанк.
        if b["section_title"] != cur_title and cur_title:
            flush()
            cur_title = b["section_title"]
            cur_section = b["section_path"]
        if not cur_section:
            cur_section = b["section_path"]
            cur_title = b["section_title"]

        if buf_len + len(text) + 1 > max_chars and buf:
            flush()

        # Путь чанка — путь ПЕРВОГО пункта, а не раздела.
        # Без этого цитата из п. 2.1 подписывалась «п. 2».
        if not buf and b["kind"] == "item" and b["section_path"]:
            cur_section = b["section_path"]

        # Пункт с номером сохраняем с номером: это основание цитаты.
        if b["kind"] == "item" and _ITEM.match(text) is None:
            label = b["section_path"]
            # Если текст уже начинается с номера — не дублируем.
            if not re.match(rf"^{re.escape(label)}\.", text):
                text = f"{label}. {text}"
        buf.append(text)
        buf_len += len(text) + 1

    flush()
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser(description="Индексация реальных регламентов")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--model-dir", default="/opt/kb/models/rubert-tiny2-int8")
    ap.add_argument("--out", default="reports/index_real.json")
    args = ap.parse_args()

    import psycopg

    if Document is None:
        print("нужен python-docx", file=sys.stderr)
        return 2

    src = Path(args.src)
    files = sorted(src.glob("*.docx"))
    if not files:
        print(f"нет DOCX в {src}", file=sys.stderr)
        return 2

    enc = Encoder(args.model_dir)
    print(f"модель загружена, документов: {len(files)}")

    results = []
    with psycopg.connect(args.dsn, autocommit=False) as conn:
        cur = conn.cursor()
        for path in files:
            title, blocks = parse_real_docx(path)
            chunks = chunk_blocks(title, blocks)
            # Сопоставление владельца по ПРЕФИКСУ: title обрезается по
            # первому предложению, а ключи в OWNERS — полные названия.
            # Точное равенство не находило 2 из 4 документов.
            owner: dict[str, str] = {}
            for key, val in OWNERS.items():
                if title.startswith(key) or key.startswith(title):
                    owner = val
                    break
            # doc_key: транслитерация не нужна, используем имя файла без версии.
            doc_key = "real_" + hashlib.sha256(
                path.stem.encode()
            ).hexdigest()[:12]
            sha = hashlib.sha256(path.read_bytes()).hexdigest()

            print(f"\n{path.name}")
            print(f"  title: {title[:60]}")
            print(f"  blocks: {len(blocks)}  chunks: {len(chunks)}")
            print(f"  owner: {owner.get('owner_role', 'не задан')}")

            cur.execute("""
                INSERT INTO documents
                    (doc_key, title, version, doc_type, owner_unit, owner_role,
                     owner_contact, source_path, content_sha256, valid_from,
                     valid_to, access_groups)
                VALUES (%s, %s, 'актуальная', 'regulation', %s, %s, %s, %s, %s,
                        current_date, NULL, ARRAY['all'])
                ON CONFLICT (doc_key, version) DO UPDATE SET
                    title = EXCLUDED.title,
                    owner_unit = EXCLUDED.owner_unit,
                    owner_role = EXCLUDED.owner_role,
                    owner_contact = EXCLUDED.owner_contact,
                    source_path = EXCLUDED.source_path,
                    content_sha256 = EXCLUDED.content_sha256,
                    indexed_at = now()
                RETURNING id
            """, (
                doc_key, title, owner.get("owner_unit"), owner.get("owner_role"),
                owner.get("owner_contact"), str(path), sha,
            ))
            doc_id = cur.fetchone()[0]

            # Переиндексация атомарна: удаляем старые чанки документа.
            cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))

            texts = [c["body"] for c in chunks]
            vecs = enc.encode(texts) if texts else []

            for i, (ch, vec) in enumerate(zip(chunks, vecs)):
                embed_text = (
                    f"{title}. Раздел {ch['section_path']} "
                    f"{ch['section_title']}. {ch['body']}"
                )
                cur.execute("""
                    INSERT INTO chunks
                        (document_id, ordinal, section_path, section_title,
                         h1_path, body, embed_text, chunk_type, char_len,
                         embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'item', %s, %s::halfvec)
                    RETURNING id
                """, (
                    doc_id, i, ch["section_path"], ch["section_title"],
                    ch["section_path"].split(".")[0], ch["body"], embed_text,
                    ch["char_len"], to_pgvector(vec),
                ))
                chunk_id = cur.fetchone()[0]

                # Предложения для реранка.
                from app.ingest.sentence_split import split_sentences
                sents = [s for s in split_sentences(ch["body"]) if len(s) >= 15]
                if not sents:
                    sents = [ch["body"]]
                svecs = enc.encode(sents)
                for si, (st, sv) in enumerate(zip(sents, svecs)):
                    cur.execute("""
                        INSERT INTO chunk_sentences (chunk_id, seq, text, embedding)
                        VALUES (%s, %s, %s, %s::halfvec)
                        ON CONFLICT (chunk_id, seq) DO UPDATE SET
                            text = EXCLUDED.text, embedding = EXCLUDED.embedding
                    """, (chunk_id, si, st, to_pgvector(sv)))

            conn.commit()
            results.append({
                "file": path.name, "title": title, "doc_key": doc_key,
                "blocks": len(blocks), "chunks": len(chunks),
                "owner_role": owner.get("owner_role"),
            })

        cur.execute("SELECT * FROM kb_refresh_activity()")
        conn.commit()
        cur.execute("""
            SELECT count(*) FILTER (WHERE is_active), count(*)
            FROM chunks
        """)
        active, total = cur.fetchone()
        cur.close()

    print(f"\nитого: {len(results)} документов")
    print(f"чанков: активных {active} из {total}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"documents": results,
                               "active_chunks": active,
                               "total_chunks": total},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"отчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())