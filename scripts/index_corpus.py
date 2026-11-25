"""Индексатор: DOCX -> БД (документы, чанки, векторы, предложения).

Идемпотентность: если content_sha256 документа не изменился — пропускаем.
Это важно, потому что переиндексация 20 документов стоит ~20 с CPU,
а в проде корпус больше и запускается по расписанию.

Порядок работы:
  1. Разобрать DOCX (парсер по контракту).
  2. Разбить на чанки (по пунктам, со склейкой коротких).
  3. Посчитать эмбеддинг каждого чанка по embed_text (с контекстом раздела).
  4. Разбить каждый чанк на предложения и посчитать их векторы —
     это основа реранка без вызова модели на запросе.
  5. Записать всё одной транзакцией на документ.

Почему батчами: ONNX-сессия на батче 32 работает в разы быстрее, чем
по одному вызову на чанк. На 61 чанке разница невелика, на 20k — решающая.

RAM: модель 29.5 МБ + сессия ~150 МБ. Пик на корпусе 20 документов: ~250 МБ.
Запускается отдельным процессом (kb-indexer.service, MemoryMax=600M),
в API не встраивается.

Запуск:
    python scripts/index_corpus.py --corpus data/corpus --dsn "..."
    python scripts/index_corpus.py --corpus data/corpus --reindex
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest.chunker import Chunk, chunk_document, chunk_stats  # noqa: E402
from app.ingest.docx_parser import ParsedDocument, parse_docx  # noqa: E402
from app.ingest.sentence_split import split_sentences  # noqa: E402
from app.search.encoder import Encoder, to_pgvector  # noqa: E402
from app.utils.vec import EMB_DIM  # noqa: E402

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]


def file_sha256(path: Path) -> str:
    """Хэш содержимого файла — для идемпотентности переиндексации."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def sentences_for(chunks: list[Chunk], min_len: int = 15) -> list[tuple[int, int, str]]:
    """Собирает (ordinal чанка, seq, текст предложения) для всех чанков.

    min_len отсекает обрывки: «См. п. 4.2.» не должен попасть в индекс
    предложений, иначе max-cosine при реранке будет завышен на служебных
    фразах, которые встречаются во всех документах.
    """
    out: list[tuple[int, int, str]] = []
    for ch in chunks:
        sents = [s for s in split_sentences(ch.body) if len(s) >= min_len]
        if not sents:
            # Чанк без пригодных предложений: берём тело целиком, иначе
            # у него не будет ни одного вектора и реранк его потеряет.
            sents = [ch.body]
        for i, s in enumerate(sents):
            out.append((ch.ordinal, i, s))
    return out


def upsert_document(
    conn,
    doc: ParsedDocument,
    sha: str,
    *,
    doc_type: str = "regulation",
    access_groups: list[str] | None = None,
) -> int:
    """Создаёт или обновляет строку documents. Возвращает id."""
    meta = doc.meta
    cur = conn.cursor()

    # valid_from обязателен в схеме. Если в документе даты нет — берём
    # дату файла: без неё документ не попадёт в выборку актуальных,
    # а молча пропустить документ хуже, чем поставить приблизительную дату.
    valid_from = meta.get("valid_from")
    if not valid_from:
        valid_from = "2025-01-01"
    valid_to = meta.get("valid_to")
    version = meta.get("version", "1.0")

    cur.execute("""
        INSERT INTO documents (doc_key, title, version, doc_type, owner_unit,
                               owner_role, owner_contact, source_path,
                               content_sha256, valid_from, valid_to, access_groups)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::date, %s::date, %s)
        ON CONFLICT (doc_key, version) DO UPDATE SET
            title = EXCLUDED.title,
            doc_type = EXCLUDED.doc_type,
            owner_unit = EXCLUDED.owner_unit,
            owner_role = EXCLUDED.owner_role,
            owner_contact = EXCLUDED.owner_contact,
            source_path = EXCLUDED.source_path,
            content_sha256 = EXCLUDED.content_sha256,
            valid_from = EXCLUDED.valid_from,
            valid_to = EXCLUDED.valid_to,
            access_groups = EXCLUDED.access_groups,
            indexed_at = now()
        RETURNING id
    """, (
        doc.doc_key, doc.title, version, doc_type,
        meta.get("owner_unit"), meta.get("owner_role"),
        meta.get("owner_contact"), doc.source_path, sha,
        valid_from, valid_to,
        access_groups or ["all"],
    ))
    doc_id = cur.fetchone()[0]
    cur.close()
    return doc_id


def index_document(
    conn,
    enc: Encoder,
    path: Path,
    *,
    doc_type: str = "regulation",
    force: bool = False,
) -> dict:
    """Индексирует один документ. Возвращает статистику."""
    doc = parse_docx(path)
    sha = file_sha256(path)

    cur = conn.cursor()
    cur.execute(
        "SELECT id, content_sha256 FROM documents WHERE doc_key=%s AND version=%s",
        (doc.doc_key, doc.meta.get("version", "1.0")),
    )
    row = cur.fetchone()
    cur.close()

    if row and row[1] == sha and not force:
        return {"file": path.name, "status": "skipped", "reason": "sha256 не изменился"}

    chunks = chunk_document(doc)
    if not chunks:
        return {"file": path.name, "status": "error", "reason": "нет чанков"}

    # Эмбеддинги: сначала все чанки одним батчем, затем все предложения.
    t0 = time.perf_counter()
    chunk_vecs = enc.encode([c.embed_text for c in chunks])
    sents = sentences_for(chunks)
    sent_vecs = enc.encode([s[2] for s in sents]) if sents else []
    embed_ms = (time.perf_counter() - t0) * 1000

    doc_id = upsert_document(conn, doc, sha, doc_type=doc_type)

    # Чанки документа перезаписываем целиком: ON DELETE CASCADE уберёт
    # и chunk_sentences. Частичное обновление сложнее и не нужно —
    # документ индексируется атомарно.
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE document_id=%s", (doc_id,))

    ordinal_to_id: dict[int, int] = {}
    for ch, vec in zip(chunks, chunk_vecs):
        cur.execute("""
            INSERT INTO chunks (document_id, ordinal, section_path, section_title,
                                h1_path, body, embed_text, chunk_type, char_len,
                                table_index, row_index, parent_ordinal, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::halfvec)
            RETURNING id
        """, (
            doc_id, ch.ordinal, ch.section_path, ch.section_title,
            ch.h1_path, ch.body, ch.embed_text, ch.chunk_type, ch.char_len,
            ch.table_index, ch.row_index, ch.parent_ordinal, to_pgvector(vec),
        ))
        ordinal_to_id[ch.ordinal] = cur.fetchone()[0]

    for (chunk_ordinal, seq, text), vec in zip(sents, sent_vecs):
        cid = ordinal_to_id.get(chunk_ordinal)
        if cid is None:
            continue
        cur.execute("""
            INSERT INTO chunk_sentences (chunk_id, seq, text, embedding)
            VALUES (%s, %s, %s, %s::halfvec)
            ON CONFLICT (chunk_id, seq) DO UPDATE SET
                text = EXCLUDED.text, embedding = EXCLUDED.embedding
        """, (cid, seq, text, to_pgvector(vec)))

    # Синхронизируем is_active чанков с документом: иначе новый чанк
    # у недействительной версии попадёт в поиск.
    cur.execute("""
        UPDATE chunks SET is_active = d.is_active
        FROM documents d WHERE d.id = chunks.document_id AND d.id = %s
    """, (doc_id,))
    conn.commit()
    cur.close()

    return {
        "file": path.name,
        "status": "indexed",
        "doc_key": doc.doc_key,
        "version": doc.meta.get("version"),
        "chunks": len(chunks),
        "sentences": len(sents),
        "embed_ms": round(embed_ms, 1),
        "stats": chunk_stats(chunks),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Индексация корпуса DOCX")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--model-dir", default="/opt/kb/models/rubert-tiny2-int8")
    ap.add_argument("--reindex", action="store_true",
                    help="переиндексировать, даже если sha256 не изменился")
    ap.add_argument("--out", default="reports/index_report.json")
    args = ap.parse_args()

    if psycopg is None:
        print("Нужен psycopg", file=sys.stderr)
        return 2

    corpus = Path(args.corpus)
    files = sorted(corpus.glob("*.docx"))
    if not files:
        print(f"нет DOCX в {corpus}", file=sys.stderr)
        return 2

    print(f"модель: {args.model_dir}")
    enc = Encoder(args.model_dir)
    print(f"документов: {len(files)}")

    manifest_path = corpus / "manifest.json"
    doc_types: dict[str, str] = {}
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        for d in man.get("documents", []):
            doc_types[d["file"]] = d.get("doc_type", "regulation")

    results = []
    t_start = time.perf_counter()
    with psycopg.connect(args.dsn, autocommit=False) as conn:
        for path in files:
            try:
                res = index_document(
                    conn, enc, path,
                    doc_type=doc_types.get(path.name, "regulation"),
                    force=args.reindex,
                )
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                res = {"file": path.name, "status": "error",
                       "reason": str(exc).splitlines()[0][:160]}
            results.append(res)
            mark = {"indexed": "OK ", "skipped": "SKIP", "error": "FAIL"}.get(
                res["status"], "??")
            extra = (f"чанков={res.get('chunks')} предложений={res.get('sentences')} "
                     f"эмбеддинг={res.get('embed_ms')}мс"
                     if res["status"] == "indexed"
                     else res.get("reason", ""))
            print(f"  [{mark}] {path.name:48s} {extra}")

        # Пересчёт актуальности после индексации: документы с valid_to
        # в прошлом должны немедленно уйти из поиска.
        cur = conn.cursor()
        cur.execute("SELECT * FROM kb_refresh_activity()")
        docs_changed, chunks_changed = cur.fetchone()
        conn.commit()
        cur.close()

    elapsed = time.perf_counter() - t_start
    ok = sum(1 for r in results if r["status"] == "indexed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "error")

    print(f"\nитого: {ok} проиндексировано, {skipped} пропущено, {failed} ошибок")
    print(f"актуальность пересчитана: документов={docs_changed}, чанков={chunks_changed}")
    print(f"время: {elapsed:.1f} с")

    # Сводка по БД
    with psycopg.connect(args.dsn, autocommit=False) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT count(*) FILTER (WHERE is_active) AS active,
                   count(*) AS total,
                   pg_size_pretty(pg_total_relation_size('chunks')) AS chunks_size
            FROM chunks
        """)
        a, t, size = cur.fetchone()
        cur.execute("SELECT count(*) FROM chunk_sentences")
        n_sent = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM documents WHERE is_active")
        n_docs = cur.fetchone()[0]
        cur.close()
        print(f"БД: документов активных={n_docs}, чанков активных={a}/{t}, "
              f"предложений={n_sent}, размер chunks={size}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "documents": len(files),
        "indexed": ok, "skipped": skipped, "failed": failed,
        "elapsed_s": round(elapsed, 2),
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"отчёт: {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())