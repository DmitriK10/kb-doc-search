"""Проверка схемы БД без pgvector.

Проблема: на этой машине pgvector не устанавливается (нужны права
администратора), но схема от него зависит — тип halfvec(312).

Решение: подменяем ТОЛЬКО тип векторных колонок на TEXT. Всё остальное
проверяется по-настоящему:
  - EXCLUDE «две версии одного документа не действуют одновременно»;
  - CHECK на response_type, включая 'stale' (правка 3.2);
  - generated-колонка tsv через kb_tsv;
  - дедупликация алертов (UNIQUE по cluster_id+reason+created_day);
  - PK кэша, включающий acl_signature (правка 3.1);
  - kb_refresh_activity() — пересчёт актуальности по текущей дате.

Что НЕ проверяется: оператор <=>, HNSW и halfvec_cosine_ops. Это
записывается в отчёт как skipped, а не как пройденное.

Запуск: python scripts/test_schema.py --dsn "..."
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]

from app.utils.sqlsplit import split_sql  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "db" / "migrations"
REPORTS = ROOT / "reports"

FAILS: list[str] = []
SKIPPED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail and ok else ""))
    if not ok:
        FAILS.append(name)
    return ok


def fake_vec_expr(has_vector: bool) -> str:
    """SQL-ВЫРАЖЕНИЕ вектора-заглушки, подставляемое в текст запроса.

    Почему выражение, а не параметр: psycopg передаёт параметр строкой,
    и PostgreSQL валидирует её как литерал halfvec («Vector contents must
    start with [»). Выражение же вычисляется сервером.

    С pgvector: настоящий halfvec из 312 нулей.
    Без pgvector (тип подменён на TEXT): обычный строковый литерал.
    """
    if has_vector:
        return "array_fill(0::real, ARRAY[312])::halfvec(312)"
    return "'v'"


def pgvector_available(conn) -> bool:
    """Есть ли расширение vector в этой базе."""
    cur = conn.cursor()
    try:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        return cur.fetchone() is not None
    finally:
        cur.close()


def devectorize(sql: str) -> str:
    """halfvec(312)/vector(312) -> TEXT. Только для среды БЕЗ pgvector.

    На сервере с pgvector схема применяется как есть: подмена типа скрыла бы
    именно то, что нужно проверить (halfvec, ограничения, индексы).
    """
    sql = re.sub(r"\bhalfvec\(\d+\)", "TEXT", sql)
    sql = re.sub(r"\bvector\(\d+\)", "TEXT", sql)
    sql = re.sub(r"\bhalfvec_cosine_ops\b", "text_ops", sql)
    return sql


def apply_sql(conn, sql: str) -> list[str]:
    """Применяет SQL операторами; возвращает список ошибок."""
    cur = conn.cursor()
    errors: list[str] = []
    for stmt in split_sql(sql):
        head = " ".join(stmt.split())[:90]
        # \echo и прочие psql-метакоманды через psycopg не работают
        if stmt.lstrip().startswith("\\"):
            continue
        try:
            cur.execute(stmt)
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            conn.rollback()
            errors.append(f"{head} -> {str(exc).splitlines()[0][:140]}")
    cur.close()
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="Проверка схемы без pgvector")
    ap.add_argument("--dsn", required=True)
    args = ap.parse_args()

    if psycopg is None:
        print("Нужен psycopg: pip install 'psycopg[binary]'", file=sys.stderr)
        return 2

    REPORTS.mkdir(exist_ok=True)
    conn = psycopg.connect(args.dsn, autocommit=False)

    # Есть ли pgvector — от этого зависит и способ применения схемы,
    # и набор проверок. Определяем ДО подготовки.
    has_vector = pgvector_available(conn)
    print("pgvector: " + ("ДА — схема применяется как есть"
                          if has_vector else "НЕТ — halfvec подменяется на TEXT"))

    print("\n== Подготовка ==")
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS btree_gist")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    # Чистим данные предыдущего прогона: тест должен быть повторяемым.
    cur.execute("""
        TRUNCATE answer_cache, document_changes, knowledge_alerts,
                 knowledge_owners, query_clusters, cluster_runs, query_logs,
                 chunk_sentences, chunks, documents
        RESTART IDENTITY CASCADE
    """)
    conn.commit()
    cur.close()

    print("\n== Миграции ==")
    if has_vector:
        migs = ("000_extensions.sql", "002_lexeme.sql", "003_synonyms.sql",
                "001_schema.sql")
    else:
        SKIPPED.append("000_extensions.sql (требует pgvector)")
        migs = ("002_lexeme.sql", "003_synonyms.sql", "001_schema.sql")

    for mig in migs:
        path = MIGRATIONS / mig
        raw = path.read_text(encoding="utf-8")
        sql = raw if has_vector else devectorize(raw)
        errs = apply_sql(conn, sql)
        check(f"{mig} применена", not errs, "; ".join(errs)[:400] if errs else "")

    cur = conn.cursor()

    print("\n== Таблицы созданы ==")
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """)
    tables = {r[0] for r in cur.fetchall()}
    expected = {
        "documents", "chunks", "chunk_sentences", "query_logs", "cluster_runs",
        "query_clusters", "knowledge_owners", "knowledge_alerts",
        "answer_cache", "document_changes", "term_synonyms",
    }
    missing = expected - tables
    check("все 11 таблиц созданы", not missing, f"нет: {sorted(missing)}" if missing else "")

    print("\n== Версионность: EXCLUDE не даёт двум версиям действовать ==")
    cur.execute("""
        INSERT INTO documents (doc_key, title, version, source_path, valid_from)
        VALUES ('reg_dogovora', 'Регламент согласования договоров', '2.3',
                '/docs/reg_v2.3.docx', '2024-01-01')
    """)
    conn.commit()

    try:
        cur.execute("""
            INSERT INTO documents (doc_key, title, version, source_path, valid_from)
            VALUES ('reg_dogovora', 'Регламент согласования договоров', '3.1',
                    '/docs/reg_v3.1.docx', '2026-03-01')
        """)
        conn.commit()
        check("вторая действующая версия отклонена", False,
              "вставка прошла — EXCLUDE не работает")
        cur.execute("DELETE FROM documents WHERE version = '3.1'")
        conn.commit()
    except psycopg.errors.ExclusionViolation:
        conn.rollback()
        check("вторая действующая версия отклонена (ExclusionViolation)", True,
              "это и решает боль «устаревший регламент» на уровне БД")

    cur.execute("""
        UPDATE documents SET valid_to = '2026-03-01'
        WHERE doc_key = 'reg_dogovora' AND version = '2.3'
    """)
    conn.commit()
    try:
        cur.execute("""
            INSERT INTO documents (doc_key, title, version, source_path, valid_from)
            VALUES ('reg_dogovora', 'Регламент согласования договоров', '3.1',
                    '/docs/reg_v3.1.docx', '2026-03-01')
        """)
        conn.commit()
        check("после закрытия v2.3 новая версия вставляется", True)
    except psycopg.errors.ExclusionViolation:
        conn.rollback()
        check("после закрытия v2.3 новая версия вставляется", False,
              "ExclusionViolation — границы диапазонов пересекаются")

    print("\n== Актуальность: kb_refresh_activity ==")
    cur.execute("""
        INSERT INTO documents (doc_key, title, version, source_path, valid_from, valid_to)
        VALUES ('old_doc', 'Старый приказ', '1.0', '/docs/old.docx',
                '2020-01-01', '2021-01-01')
    """)
    conn.commit()
    cur.execute("SELECT * FROM kb_refresh_activity()")
    changed = cur.fetchone()
    conn.commit()
    cur.execute("""
        SELECT version, is_active FROM documents
        WHERE doc_key = 'reg_dogovora' ORDER BY version
    """)
    states = dict(cur.fetchall())
    check("v2.3 неактивна (valid_to в прошлом)", states.get("2.3") is False,
          f"2.3 -> {states.get('2.3')}")
    check("v3.1 активна (valid_to пуст)", states.get("3.1") is True,
          f"3.1 -> {states.get('3.1')}")
    check("функция вернула счётчик изменений", changed is not None, str(changed))

    print("\n== Чанки: generated tsv через kb_tsv ==")
    cur.execute("SELECT id FROM documents WHERE doc_key='reg_dogovora' AND version='3.1'")
    doc_id = cur.fetchone()[0]

    # Вектор-заглушка: на сервере с pgvector колонка имеет тип halfvec(312),
    # и строковый литерал не пройдёт валидацию типа. Выражение подставляется
    # в текст запроса, потому что параметр psycopg передаётся строкой и
    # PostgreSQL валидирует его как литерал halfvec.
    cur.execute("""
        INSERT INTO chunks (document_id, ordinal, section_path, section_title,
                            h1_path, body, embed_text, chunk_type, char_len, embedding)
        VALUES (%s, 0, '3.2', 'Подписание договоров', '3',
                'Агентский договор подписывает финансовый директор при сумме свыше 1 000 000 рублей.',
                'Регламент. Раздел 3 Подписание договоров. Пункт 3.2. Агентский договор...',
                'item', 89, {VEC})
    """.format(VEC=fake_vec_expr(has_vector)), (doc_id,))
    conn.commit()
    cur.execute("SELECT tsv FROM chunks WHERE document_id=%s AND ordinal=0", (doc_id,))
    tsv = cur.fetchone()[0]
    check("generated tsv заполнен",
          tsv is not None and "агентск" in tsv and "подписыва" in tsv, str(tsv)[:80])
    check("вес заголовка A присутствует", "A" in tsv, str(tsv)[:80])
    check("вес тела B присутствует", "B" in tsv, str(tsv)[:80])

    cur.execute("SELECT is_active FROM chunks WHERE document_id=%s AND ordinal=0", (doc_id,))
    check("новый чанк активен", cur.fetchone()[0] is True)

    print("\n== CHECK: response_type принимает 'stale' ==")
    cur.execute("""
        INSERT INTO query_logs (query_hmac, query_len, embedding, response_type,
                                confidence, latency_ms, user_id_hash)
        VALUES (repeat('a',64), 20, {VEC}, 'stale', 0.7, 45, repeat('b',64))
    """.format(VEC=fake_vec_expr(has_vector)))
    conn.commit()
    check("response_type='stale' принимается", True,
          "без этого B1 ломал бы запись в журнал")

    try:
        cur.execute("""
            INSERT INTO query_logs (query_hmac, query_len, embedding, response_type,
                                    confidence, latency_ms, user_id_hash)
            VALUES (repeat('c',64), 20, {VEC}, 'bogus', 0.7, 45, repeat('d',64))
        """.format(VEC=fake_vec_expr(has_vector)))
        conn.commit()
        check("неизвестный response_type отклонён", False, "CHECK не сработал")
    except psycopg.errors.CheckViolation:
        conn.rollback()
        check("неизвестный response_type отклонён (CheckViolation)", True)

    print("\n== Алерты: дедупликация за день ==")
    cur.execute("""
        INSERT INTO cluster_runs (window_from, k, n_vectors)
        VALUES (now() - interval '7 days', 5, 60) RETURNING run_id
    """)
    run_id = cur.fetchone()[0]
    cur.execute("""
        INSERT INTO query_clusters (run_id, stable_key, centroid, size,
                                    refusal_share, useless_share, zone, label)
        VALUES (%s, 'doverennosti', {VEC}, 9, 0.8, 0.1, 'red', 'Доверенности')
        RETURNING id
    """.format(VEC=fake_vec_expr(has_vector)), (run_id,))
    cluster_id = cur.fetchone()[0]
    conn.commit()

    cur.execute("""
        INSERT INTO knowledge_alerts (cluster_id, reason, payload)
        VALUES (%s, 'refusals_3', '{}'::jsonb)
    """, (cluster_id,))
    conn.commit()
    try:
        cur.execute("""
            INSERT INTO knowledge_alerts (cluster_id, reason, payload)
            VALUES (%s, 'refusals_3', '{}'::jsonb)
        """, (cluster_id,))
        conn.commit()
        check("повторный алерт за день отклонён", False, "дубль прошёл")
    except psycopg.errors.UniqueViolation:
        conn.rollback()
        check("повторный алерт за день отклонён (UniqueViolation)", True,
              "владелец не получит 24 одинаковых сообщения в сутки")

    print("\n== Кэш: PK включает acl_signature ==")
    cur.execute("""
        INSERT INTO answer_cache (cache_key, acl_signature, payload,
                                  index_generation, expires_at)
        VALUES (repeat('e',64), 'hr', '{}'::jsonb, 1, now() + interval '1 day')
    """)
    conn.commit()
    try:
        cur.execute("""
            INSERT INTO answer_cache (cache_key, acl_signature, payload,
                                      index_generation, expires_at)
            VALUES (repeat('e',64), 'all', '{}'::jsonb, 1, now() + interval '1 day')
        """)
        conn.commit()
        check("кэш разделяется по группам доступа", True,
              "вопрос HR и обычного сотрудника — разные записи")
    except psycopg.errors.UniqueViolation:
        conn.rollback()
        check("кэш разделяется по группам доступа", False,
              "PK не включает acl_signature — утечка между группами")

    cur.execute("SELECT count(*) FROM answer_cache WHERE cache_key = repeat('e',64)")
    check("для одного вопроса две записи в разных группах",
          cur.fetchone()[0] == 2)

    print("\n== Векторная часть ==")
    if has_vector:
        vec = "[" + ",".join(["0.01"] * 312) + "]"
        cur.execute("""
            SELECT id, 1 - (embedding <=> %s::halfvec(312)) AS cos
            FROM chunks WHERE ordinal = 0
            ORDER BY embedding <=> %s::halfvec(312)
        """, (vec, vec))
        rows = cur.fetchall()
        check("оператор <=> на halfvec работает", len(rows) >= 1,
              "косинус=%.4f" % rows[0][1] if rows else "")

        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_hnsw_test
                ON chunks USING hnsw (embedding halfvec_cosine_ops)
            """)
            conn.commit()
            cur.execute("SELECT indexname FROM pg_indexes "
                        "WHERE indexname='chunks_hnsw_test'")
            check("HNSW по halfvec_cosine_ops создан", cur.fetchone() is not None)
        except Exception as exc:
            conn.rollback()
            check("HNSW по halfvec_cosine_ops создан", False,
                  str(exc).splitlines()[0][:120])
        cur.execute("DROP INDEX IF EXISTS chunks_hnsw_test")
        conn.commit()

        # Полнотекстовый поиск по настоящей таблице chunks
        cur.execute("""
            SELECT section_path FROM chunks
            WHERE tsv @@ kb_tsquery(ARRAY['подписывает','агентский'])
            LIMIT 3
        """)
        check("FTS по chunks находит пункт про агентский договор",
              len(cur.fetchall()) >= 1)
    else:
        SKIPPED.append("halfvec / <=> / HNSW (нет pgvector на этой машине)")
        print("  [SKIP] halfvec, <=>, HNSW — требуют pgvector")

    # Уборка: оставляем схему, чистим данные, чтобы следующий запуск был чистым.
    cur.execute("""
        TRUNCATE answer_cache, document_changes, knowledge_alerts,
                 knowledge_owners, query_clusters, cluster_runs, query_logs,
                 chunk_sentences, chunks, documents
        RESTART IDENTITY CASCADE
    """)
    conn.commit()
    cur.close()
    conn.close()

    print("\n" + "=" * 60)
    print(f"Провалено: {len(FAILS)}")
    if FAILS:
        print("  " + "; ".join(FAILS))
    if SKIPPED:
        print("Пропущено: " + "; ".join(SKIPPED))

    out = REPORTS / f"schema_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({
        "failed": FAILS, "skipped": SKIPPED,
        "at": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Отчёт: {out}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())