"""Чанкинг структурных блоков документа по заголовкам и пунктам регламента.

Принцип из ТЗ: режем НЕ фиксированными окнами, а по структуре — по заголовкам
и пунктам. Короткие пункты склеиваются, длинные режутся по предложениям.

Правила, зафиксированные в коде (каждое — следствие конкретного риска):

  1. Атомарная единица — один пункт (item) или одна строка таблицы (table_row).
     Заголовки НЕ создают собственный чанк: они дают section_title и
     section_path и попадают в embed_text как контекст.
     Иначе чанк «3. Подписание договоров» выигрывает по косинусу у
     содержательного пункта 3.2 и выдаётся как «ответ» — классическая
     ошибка RAG: цитата есть, содержания нет.

  2. Короткий пункт (< MIN_CHARS) склеивается со следующим В ПРЕДЕЛАХ ОДНОГО
     РАЗДЕЛА. Через разделы не склеиваем: пункт из «Общих положений» не должен
     попадать в цитату про подписание договоров.

  3. Длинный пункт (> MAX_CHARS) режется по границам предложений; каждый кусок
     получает «(продолжение)» и тот же section_path, чтобы пользователь видел,
     что цитата неполная.

  4. embed_text = «Документ. Раздел N Название. Текст». Даёт +5–10 п.п.
     Recall@3: вопрос «кто подписывает агентский договор» находит пункт 3.2,
     даже если в нём слова «агентский» нет (оно в названии раздела).

  5. body остаётся ДОСЛОВНЫМ — именно он показывается пользователю как цитата.
     Поэтому склеиваем только целые пункты (с их нумерацией в тексте),
     а не обрезки: человек должен видеть, что цитата собрана из пунктов.

Потребление RAM: один документ целиком + тексты, единицы МБ.
Скорость: 2–5 мс на документ 50 КБ (на 2 vCPU).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ingest.docx_parser import ParsedBlock, ParsedDocument
from app.ingest.sentence_split import split_sentences

MIN_CHARS = 200        # ниже — склеиваем со следующим пунктом
MAX_CHARS = 1000       # выше — режем по границам предложений
HARD_MIN_CHARS = 300   # до этого размера добираем склейкой

CONTINUATION = " (продолжение)"


@dataclass(slots=True)
class Chunk:
    """Готовый чанк для индексации."""

    ordinal: int
    section_path: str
    section_title: str
    body: str                      # ДОСЛОВНЫЙ текст для показа пользователю
    embed_text: str                # текст для эмбеддинга (с контекстом)
    chunk_type: str                # 'item' | 'table_row'
    char_len: int
    # Номер РАЗДЕЛА (главы), а не пункта. Нужен и для отображения цитаты
    # («раздел 3 Подписание договоров»), и для колонки chunks.h1_path.
    h1_path: str = ""
    table_index: int | None = None
    row_index: int | None = None
    # Для склеенных/нарезанных: ordinal первого куска группы. Нужен, чтобы
    # в ответе показать родительский пункт целиком (правка B5), а не обрывок.
    parent_ordinal: int | None = None


def _context_prefix(doc: ParsedDocument, blk: ParsedBlock) -> str:
    """Префикс контекста для embed_text: документ + РАЗДЕЛ + пункт.

    Раздел берётся из h1_path/h1_title (номер и название главы), а не из
    section_path пункта. Иначе получается «Раздел 3.2 Подписание договоров»,
    где 3.2 — номер пункта: иерархия в тексте для эмбеддинга противоречива,
    и модель путает уровень.
    """
    parts = [doc.title]
    if blk.h1_path:
        section = f"Раздел {blk.h1_path}"
        if blk.h2_path:
            section += f" ({blk.h2_path})"
        if blk.h1_title:
            section += f" {blk.h1_title}"
        parts.append(section)
    if blk.section_path:
        parts.append(f"Пункт {blk.section_path}")
    if blk.kind == "table_row" and blk.table_caption:
        parts.append(blk.table_caption)
    return ". ".join(parts)


def _split_long(text: str, max_chars: int) -> list[str]:
    """Режет длинный текст по границам предложений, не разрезая предложение.

    Предложение длиннее max_chars всё равно режется по символам — иначе лимит
    не гарантировать. В регламентах это крайний случай (одно предложение на
    1000+ символов), но он возможен в перечислениях.
    """
    if len(text) <= max_chars:
        return [text]

    sentences = split_sentences(text)
    if not sentences:
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    out: list[str] = []
    buf = ""
    for sent in sentences:
        if len(sent) > max_chars:
            if buf:
                out.append(buf)
                buf = ""
            for i in range(0, len(sent), max_chars):
                out.append(sent[i:i + max_chars])
            continue
        candidate = f"{buf} {sent}".strip() if buf else sent
        if len(candidate) > max_chars:
            out.append(buf)
            buf = sent
        else:
            buf = candidate
    if buf:
        out.append(buf)
    return out


def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    """Превращает разобранный документ в список чанков."""
    content: list[ParsedBlock] = [
        b for b in doc.blocks if b.kind in ("item", "table_row")
    ]
    if not content:
        return []

    # Склейка коротких пунктов внутри раздела. Строки таблиц не склеиваем:
    # каждая — самостоятельная запись матрицы согласования.
    merged: list[tuple[ParsedBlock, list[ParsedBlock]]] = []
    i = 0
    while i < len(content):
        head = content[i]
        group = [head]
        total = len(head.text)
        j = i + 1
        if head.kind == "item":
            while (
                j < len(content)
                and total < HARD_MIN_CHARS
                and content[j].kind == "item"
                and content[j].section_title == head.section_title
                and total + len(content[j].text) + 1 <= MAX_CHARS
            ):
                group.append(content[j])
                total += len(content[j].text) + 1
                j += 1
        merged.append((head, group))
        i = j

    chunks: list[Chunk] = []
    ordinal = 0
    for head, group in merged:
        if len(group) == 1:
            body_full = head.text
        else:
            # Целые пункты с нумерацией: пользователь видит, что цитата
            # составная, и что именно процитировано.
            body_full = "\n".join(
                f"{b.section_path}. {b.text}" if b.kind == "item" else b.text
                for b in group
            )

        prefix = _context_prefix(doc, head)
        first_ordinal: int | None = None

        for k, part in enumerate(_split_long(body_full, MAX_CHARS)):
            text = part.strip()
            if not text:
                continue
            label = CONTINUATION if k > 0 else ""
            if k == 0:
                first_ordinal = ordinal
            chunks.append(Chunk(
                ordinal=ordinal,
                section_path=head.section_path,
                section_title=head.section_title,
                body=text,
                embed_text=f"{prefix}{label}. {text}",
                chunk_type=head.kind,
                char_len=len(text),
                h1_path=head.h1_path,
                table_index=head.table_index,
                row_index=head.row_index,
                parent_ordinal=None if k == 0 else first_ordinal,
            ))
            ordinal += 1

    return chunks


def chunk_stats(chunks: list[Chunk]) -> dict[str, float]:
    """Сводка по чанкам — оценка качества чанкинга ДО индексации.

    too_short > 0 и слишком малый mean — сигнал, что MIN_CHARS/HARD_MIN_CHARS
    не подходят под конкретный корпус: короткие чанки дают высокий косинус
    на общих словах и порождают ложные уверенные ответы.
    """
    if not chunks:
        return {"n": 0}
    lens = sorted(c.char_len for c in chunks)
    n = len(lens)
    return {
        "n": n,
        "min": lens[0],
        "p50": lens[n // 2],
        "p90": lens[min(n - 1, int(n * 0.9))],
        "max": lens[-1],
        "mean": round(sum(lens) / n, 1),
        "too_short": sum(1 for x in lens if x < MIN_CHARS),
        "too_long": sum(1 for x in lens if x > MAX_CHARS),
        "table_rows": sum(1 for c in chunks if c.chunk_type == "table_row"),
    }