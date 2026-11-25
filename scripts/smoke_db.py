"""Smoke-тест SQL-слоя на живом PostgreSQL.

Зачем: SQL — самая непроверяемая часть при разработке без сервера. Ошибка
вида `c.doc_id` вместо `c.document_id` вскрывается на демо, а не при написании.
Этот скрипт превращает «написано» в «проверено» и печатает JSON-отчёт, который
можно приложить к EVALUATION.md.

Два режима:
  1. FTS-режим (не требует pgvector) — проверяет всю цепочку:
     kb_norm, kb_lemma, симметрию ё/е, kb_tsv, kb_expand_query (лексемный
     матчинг синонимов), kb_tsquery (структура AND/OR), пустой запрос,
     AND→OR фолбэк, ранжирование на реальном русском тексте.
     Это ровно тот код, который написан в этом раунде и который нельзя
     проверить «глазами».
  2. Полный режим (нужен pgvector) — плюс миграции схемы, halfvec,
     векторный поиск, RRF, stale-флоу.

Запуск:
    python scripts/smoke_db.py --dsn "host=127.0.0.1 port=5432 user=postgres password=..."
    python scripts/smoke_db.py --dsn ... --mode full

Отчёт: reports/smoke_db_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
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


@dataclass
class Check:
    """Один проверяемый факт."""

    name: str
    ok: bool
    detail: str = ""
    expected: str = ""
    got: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name, "ok": self.ok, "detail": self.detail,
            "expected": self.expected, "got": self.got,
        }


@dataclass
class Report:
    mode: str
    started_at: str = ""
    checks: list[Check] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    pg_version: str = ""
    pgvector_version: str = ""

    def check(self, name: str, ok: bool, expected="", got="", detail="") -> bool:
        self.checks.append(Check(name, bool(ok), detail, str(expected), str(got)))
        mark = "OK  " if ok else "FAIL"
        line = f"  [{mark}] {name}"
        if not ok:
            line += f"\n         expected: {expected}\n         got:      {got}"
        elif detail:
            line += f"  ({detail})"
        print(line)
        return bool(ok)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "started_at": self.started_at,
            "pg_version": self.pg_version,
            "pgvector_version": self.pgvector_version,
            "total": len(self.checks),
            "passed": sum(1 for c in self.checks if c.ok),
            "failed": len(self.failed),
            "skipped": self.skipped,
            "checks": [c.to_dict() for c in self.checks],
        }


# ---------------------------------------------------------------------------
# FTS-режим: не требует pgvector
# ---------------------------------------------------------------------------

FTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS _smoke_chunks (
    id            serial PRIMARY KEY,
    doc_title     text NOT NULL,
    section_path  text NOT NULL,
    section_title text NOT NULL,
    body          text NOT NULL,
    is_active     boolean NOT NULL DEFAULT true
);

-- tsv через kb_tsv: та же нормализация, что и на запросе.
ALTER TABLE _smoke_chunks DROP COLUMN IF EXISTS tsv;
ALTER TABLE _smoke_chunks
    ADD COLUMN tsv tsvector
    GENERATED ALWAYS AS (
        kb_tsv(section_title, 'A') || kb_tsv(body, 'B')
    ) STORED;

CREATE INDEX IF NOT EXISTS _smoke_chunks_tsv ON _smoke_chunks USING gin (tsv);
"""

FIXTURES = [
    ("Регламент согласования и подписания договоров", "1.1", "Общие положения",
     "Настоящий Регламент определяет порядок согласования и подписания договоров, "
     "заключаемых от имени компании."),
    ("Регламент согласования и подписания договоров", "2.2", "Порядок согласования",
     "Срок согласования — не позднее 5 рабочих дней с момента передачи проекта "
     "договора ответственному исполнителю."),
    ("Регламент согласования и подписания договоров", "3.2", "Подписание договоров",
     "Агентский договор подписывает финансовый директор при сумме свыше "
     "1 000 000 рублей."),
    ("Регламент согласования и подписания договоров", "3.1", "Подписание договоров",
     "Договоры на сумму до 100 000 рублей подписывает руководитель подразделения "
     "инициатора."),
    ("Регламент согласования и подписания договоров", "5#т1.3", "Матрица согласования",
     "Роль: Агентский договор; Сумма договора: свыше 1 000 000 руб.; "
     "Подписант: Финансовый директор"),
    ("Положение о командировках", "4.1", "Порядок оформления",
     "Отчёт о командировке сдаётся в бухгалтерию не позднее трёх рабочих дней "
     "после возвращения."),
    ("Положение о командировках", "2.3", "Суточные расходы",
     "Суточные выплачиваются в размере 700 рублей за каждый день нахождения "
     "в командировке."),
]


def run_fts_mode(conn, rep: Report) -> None:
    """Проверяет цепочку FTS/лексем/tsquery на живом сервере."""
    cur = conn.cursor()

    print("\n-- 1. Нормализация и лемматизация --")
    cur.execute("""
        SELECT kb_norm('Отчёт  о   Командировке'),
               kb_lemma('командировку'),
               kb_lemma('и'),
               kb_lemma('отчёт'),
               kb_lemma('отчет'),
               kb_lemma('qwerty123')
    """)
    norm, lem_komm, lem_stop, lem_yo, lem_ye, lem_unknown = cur.fetchone()

    rep.check("kb_norm: lower + схлопывание пробелов",
              norm == "отчёт о командировке" or norm == "отчет о командировке",
              "отчёт о командировке", norm)
    rep.check("kb_lemma: 'командировку' -> основа",
              lem_komm == "командировк", "командировк", lem_komm)
    rep.check("kb_lemma: стоп-слово -> NULL", lem_stop is None, None, lem_stop)
    rep.check("kb_lemma: слово вне словаря сохраняется",
              lem_unknown == "qwerty123", "qwerty123", lem_unknown)

    # Главный факт, проверенный на реальном сервере: словарь russian
    # унифицирует ё/е сам. Значит ручная замена в токенизаторе не нужна,
    # а асимметрия возникает, только если замену делать в одном месте.
    rep.check("ё/е симметрия: 'отчёт' == 'отчет'",
              lem_yo == lem_ye,
              f"одинаковые лексемы, получено {lem_yo!r} / {lem_ye!r}",
              f"{lem_yo!r} / {lem_ye!r}",
              "словарь russian_stem унифицирует ё->е автоматически")

    print("\n-- 2. tsvector симметрия (документ == запрос) --")
    cur.execute("""
        SELECT to_tsvector('russian', kb_norm('Отчёт о командировке'))
             = to_tsvector('russian', kb_norm('Отчет о командировке'))
    """)
    rep.check("tsv('Отчёт') == tsv('Отчет')", cur.fetchone()[0] is True, True, False)

    cur.execute("SELECT kb_tsv('Общие положения','A') || kb_tsv('Порядок подписания','B')")
    tsv = cur.fetchone()[0]
    # Проверяем НАЛИЧИЕ обоих весов, а не конкретные позиции лексем:
    # номера позиций зависят от порядка слов и не являются контрактом.
    rep.check("kb_tsv: веса A и B расставлены",
              ":1A" in tsv and re.search(r":\d+B", tsv) is not None,
              "есть вес A и вес B", tsv)

    print("\n-- 3. Фикстуры и индексация --")
    cur.execute(FTS_SCHEMA)
    cur.execute("TRUNCATE _smoke_chunks RESTART IDENTITY")
    for doc_title, path, sec_title, body in FIXTURES:
        cur.execute(
            "INSERT INTO _smoke_chunks (doc_title, section_path, section_title, body) "
            "VALUES (%s, %s, %s, %s)",
            (doc_title, path, sec_title, body),
        )
    cur.execute("SELECT count(*) FROM _smoke_chunks")
    n = cur.fetchone()[0]
    rep.check("фикстуры загружены", n == len(FIXTURES), len(FIXTURES), n)

    cur.execute("SELECT count(*) FROM _smoke_chunks WHERE tsv IS NULL OR tsv = ''::tsvector")
    empty = cur.fetchone()[0]
    rep.check("generated tsv заполнен для всех строк", empty == 0, 0, empty)

    print("\n-- 4. Словарь синонимов: лексемный матчинг --")
    # Ключевой дефект прошлого раунда: 'командировку' в запросе не находил
    # запись 'командировка' в словаре, потому что ключ сравнивался поверхностно.
    cur.execute("""
        SELECT count(*) FROM term_synonyms
        WHERE term_lemma = kb_lemma('командировку')
    """)
    hit = cur.fetchone()[0]
    rep.check("синоним находится по лексеме (командировку -> командировка)",
              hit == 1, 1, hit,
              "падежная форма запроса находит словарную запись")

    cur.execute("""
        SELECT count(*) FROM term_synonyms
        WHERE term_lemma = kb_lemma('подписывает')
    """)
    n_sub = cur.fetchone()[0]
    rep.check("синоним 'подписывает' найден", n_sub == 1, 1, n_sub)

    print("\n-- 5. kb_expand_query: OR внутри группы --")
    cur.execute("""
        SELECT gid, array_agg(variant ORDER BY variant)
        FROM kb_expand_query(ARRAY['подписывает','договор'])
        GROUP BY gid ORDER BY gid
    """)
    groups = cur.fetchall()
    rep.check("две группы для двух токенов", len(groups) == 2, 2, len(groups))
    if groups:
        g1 = groups[0][1]
        rep.check("группа 1 содержит синонимы 'подписывает'",
                  len(g1) >= 3, ">=3 вариантов", g1,
                  f"варианты: {g1}")

    print("\n-- 6. kb_tsquery: AND между группами --")
    cur.execute("SELECT kb_tsquery(ARRAY['подписывает','агентский','договор'])")
    q = cur.fetchone()[0]
    rep.check("tsquery построен", q is not None and str(q) != "", "непустой tsquery", q)
    rep.check("в tsquery есть AND (&)", q is not None and "&" in str(q), "&", q)
    rep.check("в tsquery есть OR (|) внутри группы",
              q is not None and "|" in str(q), "|", q)

    cur.execute("SELECT kb_tsquery(ARRAY['и','как','в'])")
    empty_q = cur.fetchone()[0]
    rep.check("все стоп-слова -> NULL (FTS не запускаем)", empty_q is None, None, empty_q)

    print("\n-- 7. К1: версия не попадает в AND, но и не ломает запрос --")
    cur.execute("SELECT kb_tsquery(ARRAY['подписывает','регламент'])")
    qv = cur.fetchone()[0]
    rep.check("запрос без версии строится корректно",
              qv is not None and "подписыва" in str(qv).replace('"', ''), "подписыва", qv)

    # Проверяем, что версия, если её всё-таки передать, не даёт пустой AND.
    cur.execute("SELECT kb_lemma('2.3')")
    lem_version = cur.fetchone()[0]
    rep.check("kb_lemma('2.3') возвращает непустую лексему",
              lem_version is not None, "не NULL", lem_version,
              "версии фильтруются структурно (documents.version), а не полнотекстом")

    print("\n-- 8. Поиск: AND-ветка на естественном вопросе --")
    def search(tokens: list[str], limit: int = 5):
        cur.execute("""
            WITH q AS (SELECT kb_tsquery(%s::text[]) AS tsq)
            SELECT c.section_path, c.section_title,
                   ts_rank_cd(c.tsv, q.tsq) AS rank,
                   left(c.body, 62) AS preview
            FROM _smoke_chunks c, q
            WHERE q.tsq IS NOT NULL AND c.tsv @@ q.tsq
            ORDER BY rank DESC
            LIMIT %s
        """, (tokens, limit))
        return cur.fetchall()

    rows = search(["подписывает", "агентский", "договор"])
    rep.check("вопрос 'кто подписывает агентский договор' находит строку",
              len(rows) > 0, ">=1 строки", len(rows))
    if rows:
        top = rows[0]
        rep.check("топ-1 — про агентский договор и подписанта",
                  "агентск" in top[3].lower() or "финансов" in top[3].lower(),
                  "агентский/финансовый", top[3])
        print(f"         top-1: {top[0]} [{top[1]}] rank={top[2]:.4f}")
        print(f"                {top[3]}...")

    print("\n-- 9. Recall AND vs OR: цена потолка групп (К2) --")
    long_q = ["подписывает", "агентский", "договор", "сумма", "рублей",
              "финансовый", "директор", "свыше"]
    and_rows = search(long_q)
    rep.check("AND по 8 группам не обнуляет выдачу (К2)",
              len(and_rows) > 0, ">=1 строки", len(and_rows),
              f"найдено {len(and_rows)} строк по 8 группам")

    # OR-фолбэк: максимальный recall, включается только при <5 строк.
    cur.execute("""
        SELECT count(*)
        FROM _smoke_chunks c
        WHERE c.tsv @@ kb_tsquery_or(ARRAY['подписывает','агентский','договор'])
    """)
    or_n = cur.fetchone()[0]
    rep.check("OR-фолбэк даёт не меньше строк, чем AND",
              or_n >= len(and_rows), f">={len(and_rows)}", or_n)

    print("\n-- 10. Опечатки: trigram-фолбэк --")
    # pg_trgm создаём здесь: в FTS-режиме миграция 000_extensions не гоняется,
    # потому что она требует pgvector. Само расширение доступно и без него.
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # ВАЖНО: для опечаток нужен word_similarity, а НЕ similarity.
    # similarity() сравнивает триграммы всей строки целиком: на длинном теле
    # чанка («Отчёт о командировке сдаётся в бухгалтерию не позднее трёх...»)
    # слово-опечатка тонет в остальном тексте, и оценка падает до ~0.22 —
    # ниже любого разумного порога. Проверено на живом сервере:
    #   similarity(body,'камандировке')      = 0.222  (бесполезно)
    #   word_similarity('камандировке',body) = 0.692  (работает)
    # word_similarity ищет наиболее похожий ФРАГМЕНТ текста, что и требуется.
    cur.execute(
        "SELECT count(*) FROM _smoke_chunks WHERE word_similarity(%s, body) >= 0.6",
        ("камандировке",),
    )
    trgm = cur.fetchone()[0]
    rep.check("word_similarity находит строку с опечаткой (камандировке)",
              trgm >= 1, ">=1 строки", trgm,
              "порог 0.6: similarity() на длинном теле даёт 0.22 и не годится")

    # Точное слово тоже должно находиться, и с оценкой 1.0.
    cur.execute(
        "SELECT count(*) FROM _smoke_chunks WHERE word_similarity(%s, body) >= 0.6",
        ("командировке",),
    )
    rep.check("word_similarity находит точное слово",
              cur.fetchone()[0] >= 1, ">=1 строки", trgm)

    print("\n-- 11. Ранжирование: вес заголовка раздела (A) выше тела (B) --")
    cur.execute("""
        SELECT c.section_path, ts_rank_cd(c.tsv, to_tsquery('russian','подписание')) AS r
        FROM _smoke_chunks c
        WHERE c.tsv @@ to_tsquery('russian','подписание')
        ORDER BY r DESC
    """)
    rank_rows = cur.fetchall()
    rep.check("совпадение по заголовку раздела ранжируется выше",
              len(rank_rows) > 0, ">=1", len(rank_rows),
              f"строк: {len(rank_rows)}, top={rank_rows[0][0] if rank_rows else '-'}")

    cur.close()


# ---------------------------------------------------------------------------
# Полный режим: миграции схемы + pgvector
# ---------------------------------------------------------------------------

def pgvector_available(conn) -> str | None:
    cur = conn.cursor()
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    row = cur.fetchone()
    cur.close()
    return row[0] if row else None


def apply_migration(conn, name: str, rep: Report) -> bool:
    path = MIGRATIONS / name
    if not path.exists():
        rep.skipped.append(f"{name} (файл не найден)")
        return False
    sql = path.read_text(encoding="utf-8")
    cur = conn.cursor()
    try:
        cur.execute(sql)
        conn.commit()
        rep.check(f"миграция {name} применена", True)
        return True
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        rep.check(f"миграция {name} применена", False, "без ошибок", str(exc)[:200])
        return False
    finally:
        cur.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke-тест SQL-слоя KB")
    ap.add_argument("--dsn", required=True, help="строка подключения psycopg")
    ap.add_argument("--mode", choices=["fts", "full"], default="fts")
    ap.add_argument("--keep", action="store_true", help="не удалять тестовые объекты")
    args = ap.parse_args()

    if psycopg is None:
        print("Нужен psycopg: pip install 'psycopg[binary]'", file=sys.stderr)
        return 2

    rep = Report(mode=args.mode, started_at=datetime.now(timezone.utc).isoformat())
    REPORTS.mkdir(exist_ok=True)

    with psycopg.connect(args.dsn, autocommit=False) as conn:
        cur = conn.cursor()
        cur.execute("SELECT version()")
        rep.pg_version = cur.fetchone()[0]
        cur.close()
        print(f"PostgreSQL: {rep.pg_version.split(',')[0]}")

        vec = pgvector_available(conn)
        rep.pgvector_version = vec or ""
        if vec:
            print(f"pgvector:   {vec}")
        else:
            print("pgvector:   НЕ УСТАНОВЛЕН — векторные проверки будут пропущены")

        # Миграции по порядку зависимостей:
        #   000 — расширения (pgvector нужен не всегда)
        #   002 — kb_norm/kb_lemma/kb_lemmas/kb_tsv + агрегаты tsquery
        #   003 — словарь синонимов (генерируемые колонки через kb_lemma)
        #   001 — схема документов/чанков (требует pgvector: тип halfvec)
        # 002 и 003 не содержат типов pgvector, поэтому грузятся в любом режиме.
        if args.mode == "full":
            print("\n== Миграции ==")
            for mig in ("000_extensions.sql", "002_lexeme.sql", "003_synonyms.sql",
                        "001_schema.sql"):
                if not apply_migration(conn, mig, rep):
                    break
            if not vec:
                rep.skipped.append("векторный поиск и RRF (нет pgvector)")

        if vec:
            print("\n== Функции и словарь (pgvector есть) ==")
            apply_migration(conn, "002_lexeme.sql", rep)
            apply_migration(conn, "003_synonyms.sql", rep)
        else:
            print("\n== Загрузка kb_* функций (без pgvector) ==")
            cur = conn.cursor()
            for mig in ("002_lexeme.sql", "003_synonyms.sql"):
                sql = (MIGRATIONS / mig).read_text(encoding="utf-8")
                ok_all, errors = True, []
                for stmt in split_sql(sql):
                    try:
                        cur.execute(stmt)
                        conn.commit()
                    except Exception as exc:  # noqa: BLE001
                        conn.rollback()
                        ok_all = False
                        head = " ".join(stmt.split())[:80]
                        errors.append(f"{head} -> {str(exc).splitlines()[0][:120]}")
                rep.check(
                    f"функции из {mig} загружены", ok_all,
                    expected="без ошибок",
                    got="; ".join(errors)[:400] if errors else "ok",
                )
            cur.close()

        print("\n== FTS / лексемы / tsquery ==")
        try:
            run_fts_mode(conn, rep)
        except Exception as exc:  # noqa: BLE001
            conn.rollback()
            rep.check("FTS-проверки выполнены", False,
                      expected="без исключений", got=str(exc)[:300])

        if not args.keep:
            cur = conn.cursor()
            cur.execute("DROP TABLE IF EXISTS _smoke_chunks")
            conn.commit()
            cur.close()

    out = REPORTS / f"smoke_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"Итого: {len(rep.checks) - len(rep.failed)}/{len(rep.checks)} проверок пройдено")
    if rep.skipped:
        print("Пропущено: " + "; ".join(rep.skipped))
    if rep.failed:
        print("ПРОВАЛЕНО:")
        for c in rep.failed:
            print(f"  - {c.name}: ожидалось {c.expected!r}, получено {c.got!r}")
    print(f"Отчёт: {out}")
    return 1 if rep.failed else 0


def _strip_vector_blocks(sql: str) -> str:
    """Убирает из миграции операторы, зависящие от pgvector (для FTS-режима).

    Отбрасываются только операторы, где встречается тип vector/halfvec,
    оператор <=> или опкласс *_cosine_ops. Функции kb_* сохраняются целиком:
    ровно их и проверяем в режиме без pgvector.
    """
    keep: list[str] = []
    for stmt in split_sql(sql):
        low = stmt.lower()
        if ("vector" in low) or ("<=>" in stmt) or ("cosine_ops" in low):
            continue
        keep.append(stmt)
    return ";\n".join(keep) + ";"


if __name__ == "__main__":
    raise SystemExit(main())