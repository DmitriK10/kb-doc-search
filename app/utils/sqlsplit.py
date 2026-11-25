"""Корректное деление SQL на операторы.

Зачем отдельный модуль: наивный split(';') ломается на трёх вещах, и все три
встречаются в наших миграциях:

  1. Точка с запятой ВНУТРИ dollar-quoted тела функции ($$ ... $$):
     объявления переменных, END IF;, RETURN;.
  2. Точка с запятой ВНУТРИ строчного комментария:
     `-- Добавлено в раунде 3; CHECK обязан включать его` — именно на этом
     падала загрузка 001_schema.sql: оператор разрывался посреди CREATE TABLE.
  3. Точка с запятой внутри строкового литерала.

Разбор идёт посимвольно с состоянием, поэтому все три случая покрыты.
psql-метакоманды (\echo, \quit) отбрасываются: через psycopg они не работают.

Используется в scripts/smoke_db.py и scripts/test_schema.py.
"""

from __future__ import annotations

import re


# psql-метакоманды: \echo, \quit, \set и т.п. Через psycopg они
# не работают, а при склейке со следующим оператором дают синтаксическую
# ошибку. Удаляем их построчно ДО разбора.
_PSQL_META = re.compile(r"(?m)^[ \t]*\\[a-zA-Z]+.*$")


def split_sql(sql: str) -> list[str]:
    """Возвращает список SQL-операторов без завершающих ';' и без метакоманд."""
    sql = _PSQL_META.sub("", sql)
    out: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(sql)

    dollar_tag: str | None = None   # активный $tag$ ... $tag$
    in_single = False               # внутри '...'
    in_line_comment = False         # внутри -- ...
    block_depth = 0                 # вложенность /* ... */ (PG допускает вложенность)

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        # --- внутри dollar-quoted тела: всё литерально до закрывающего тега ---
        if dollar_tag is not None:
            if sql.startswith(dollar_tag, i):
                buf.append(dollar_tag)
                i += len(dollar_tag)
                dollar_tag = None
                continue
            buf.append(ch)
            i += 1
            continue

        # --- внутри строчного комментария ---
        if in_line_comment:
            buf.append(ch)
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        # --- внутри блочного комментария ---
        if block_depth > 0:
            buf.append(ch)
            if ch == "/" and nxt == "*":
                buf.append(nxt)
                block_depth += 1
                i += 2
                continue
            if ch == "*" and nxt == "/":
                buf.append(nxt)
                block_depth -= 1
                i += 2
                continue
            i += 1
            continue

        # --- внутри строкового литерала ---
        if in_single:
            buf.append(ch)
            if ch == "'":
                if nxt == "'":          # '' — экранированная кавычка
                    buf.append(nxt)
                    i += 2
                    continue
                in_single = False
            i += 1
            continue

        # --- обычный режим ---
        if ch == "-" and nxt == "-":
            buf.append(ch)
            buf.append(nxt)
            in_line_comment = True
            i += 2
            continue

        if ch == "/" and nxt == "*":
            buf.append(ch)
            buf.append(nxt)
            block_depth = 1
            i += 2
            continue

        if ch == "'":
            in_single = True
            buf.append(ch)
            i += 1
            continue

        if ch == "$":
            j = i + 1
            while j < n and (sql[j].isalnum() or sql[j] == "_"):
                j += 1
            if j < n and sql[j] == "$":
                dollar_tag = sql[i:j + 1]
                buf.append(dollar_tag)
                i = j + 1
                continue
            buf.append(ch)
            i += 1
            continue

        if ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                out.append(stmt)
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        out.append(tail)

    # Отбрасываем psql-метакоманды и пустые операторы (только комментарии).
    cleaned: list[str] = []
    for stmt in out:
        stripped = stmt.lstrip()
        if stripped.startswith("\\"):
            continue
        if _is_only_comments(stmt):
            continue
        cleaned.append(stmt)
    return cleaned


def _is_only_comments(stmt: str) -> bool:
    """True, если в операторе нет ничего, кроме комментариев и пробелов."""
    i = 0
    n = len(stmt)
    while i < n:
        ch = stmt[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "-" and i + 1 < n and stmt[i + 1] == "-":
            while i < n and stmt[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and stmt[i + 1] == "*":
            depth = 1
            i += 2
            while i < n and depth > 0:
                if stmt.startswith("/*", i):
                    depth += 1
                    i += 2
                elif stmt.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            continue
        return False
    return True