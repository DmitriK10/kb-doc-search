"""Парсер DOCX по контракту корпуса.

Контракт (сверен с эталоном reg_dogovora_v3.1.docx):
  1. Заголовок документа — стиль Title.
  2. Разделы — Heading 1, подразделы — Heading 2 (Heading 3+ тоже читаем).
  3. Шапка с метаданными — отдельный абзац сразу после заголовка.
     Поддерживаются ДВЕ формы, потому что эталон использует вторую:
       а) «Версия: 2.3 | Действует с: 01.03.2026 | Владелец: Юр. департамент»
       б) «Версия 3.1. Действует с 01.03.2026. Владелец: Юридический
           департамент (legal@company.ru).»
  4. Каждый пункт — отдельный абзац с нумерацией N.M. или N.M.K.
     Регулярка: ^(\\d+(?:\\.\\d+)*)[.)]?\\s+
  5. Таблица — обычная docx-таблица, шапка в первой строке, режется построчно.

Решение из smoke-теста: строка «Таблица 1. Пределы полномочий» не является ни
заголовком, ни пунктом. Относим её к следующей таблице как caption, а не
дропаем: подпись нужна пользователю, чтобы понять цитату из table_row.

Потребление RAM: python-docx держит разобранный документ целиком; для
регламента 100–300 КБ это единицы МБ. Парсер вызывается только индексатором
(отдельный процесс, MemoryMax=600M), в API не используется.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph

# Пункт: «4.2. Текст», «4.2 Текст», «4.2) Текст», «4.2.1. Текст».
ITEM_RE = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+(.*)$", re.DOTALL)

# Подпись таблицы: «Таблица 1. Пределы...», «Таблица 2 — ...».
TABLE_CAPTION_RE = re.compile(r"^таблица\s+(\d+)[.):\s—-]*(.*)$", re.IGNORECASE)

# --- Разбор шапки метаданных (две формы) -----------------------------------
# Форма «Ключ: значение» через | ; •
_META_PAIRS_SPLIT = re.compile(r"\s*[|•;]\s*")
_META_KV = re.compile(r"^([^:]{2,40}):\s*(.+)$")

# Форма «Версия 3.1. Действует с 01.03.2026. Владелец: ...»
_META_VERSION = re.compile(
    r"(?:версия|редакция|version|ред\.)\s*:?\s*v?(\d+(?:\.\d+)*)", re.IGNORECASE
)
_META_VALID_FROM = re.compile(
    r"(?:действует\s+с|вступает\s+в\s+силу|дата\s+введения|valid\s+from)\s*:?\s*"
    r"(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})",
    re.IGNORECASE,
)
_META_VALID_TO = re.compile(
    r"(?:действует\s+до|утрачивает\s+силу|valid\s+to)\s*:?\s*"
    r"(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})",
    re.IGNORECASE,
)
# «заменяет версию 2.3» — даёт предыдущую версию для проставления valid_to.
_META_REPLACES = re.compile(
    r"(?:заменяет|отменяет)\s+(?:версию|редакцию)\s*v?(\d+(?:\.\d+)*)", re.IGNORECASE
)
_META_OWNER = re.compile(
    r"(?:владелец(?:\s+документа)?|подразделение|owner)\s*:?\s*"
    r"([^;|]+?)(?:\s*\(|\.\s|\.$|$)",
    re.IGNORECASE,
)
_META_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
_META_ROLE = re.compile(
    r"(?:роль\s+владельца|ответственный)\s*:?\s*([^;|]+?)(?:\.\s|\.$|$)", re.IGNORECASE
)
_META_DOC_CODE = re.compile(
    r"(?:шифр|код\s+документа|doc\s*code)\s*:?\s*([^\s;|.,]+)", re.IGNORECASE
)

_META_KEYS_HINT = (
    "версия", "version", "редакция", "действует", "владелец", "подразделение",
    "роль владельца", "шифр", "код", "утвердил", "owner",
)


def _iso(d: str, m: str, y: str) -> str:
    """01.03.2026 -> 2026-03-01 (дата в ISO для колонки DATE)."""
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"


def looks_like_meta_line(text: str) -> bool:
    """Строка похожа на шапку метаданных: минимум два разных признака.

    Два независимых признака обязательны, иначе обычный пункт регламента
    («Владелец документа — Юридический департамент») будет съеден как шапка.
    """
    low = text.lower()
    hits = 0
    if _META_VERSION.search(text):
        hits += 1
    if _META_VALID_FROM.search(text) or _META_VALID_TO.search(text):
        hits += 1
    if _META_OWNER.search(text) or _META_EMAIL.search(text):
        hits += 1
    # плюс структурный признак формы «ключ: значение»
    if any(k in low for k in _META_KEYS_HINT):
        hits += 1
    return hits >= 2


def parse_meta(text: str) -> dict[str, str]:
    """Извлекает поля documents из строки шапки. Работает для обеих форм."""
    out: dict[str, str] = {}

    m = _META_VERSION.search(text)
    if m:
        out["version"] = m.group(1)

    m = _META_VALID_FROM.search(text)
    if m:
        out["valid_from"] = _iso(*m.groups())

    m = _META_VALID_TO.search(text)
    if m:
        out["valid_to"] = _iso(*m.groups())

    m = _META_REPLACES.search(text)
    if m:
        out["replaces_version"] = m.group(1)

    m = _META_OWNER.search(text)
    if m:
        out["owner_unit"] = " ".join(m.group(1).split()).strip(" .,—")

    m = _META_EMAIL.search(text)
    if m:
        out["owner_contact"] = m.group(0)

    m = _META_ROLE.search(text)
    if m:
        out["owner_role"] = " ".join(m.group(1).split()).strip(" .,—")

    m = _META_DOC_CODE.search(text)
    if m:
        out["doc_code"] = m.group(1)

    # Добираем форму «Ключ: значение», если она есть (не перетирая найденное).
    for chunk in _META_PAIRS_SPLIT.split(text):
        km = _META_KV.match(chunk.strip())
        if not km:
            continue
        k = km.group(1).strip().lower()
        v = km.group(2).strip()
        if k.startswith(("версия", "version")) and "version" not in out:
            out["version"] = v.lstrip("vV")
        elif k.startswith(("владелец", "подразделение", "owner")) and "owner_unit" not in out:
            out["owner_unit"] = v
    return out


@dataclass(slots=True)
class ParsedBlock:
    """Один структурный элемент документа в порядке чтения.

    ВАЖНО: section_path для пункта — это номер ПУНКТА («3.2»), а h1_path/h2_path
    — номера РАЗДЕЛОВ. Смешивать нельзя: в embed_text должен идти раздел
    («Раздел 3 Подписание договоров»), иначе контекст для эмбеддинга
    получается «Раздел 3.2 Подписание договоров» — номер пункта выдаётся
    за номер раздела, и модель видит противоречивую иерархию.
    """

    kind: str                      # title | h1 | h2 | item | table_row
    text: str
    section_path: str = ""         # номер пункта '4.2' | '5#т1.3' для строк таблицы
    section_title: str = ""        # ближайший заголовок: 'Подписание договоров'
    h1_path: str = ""              # номер раздела верхнего уровня: '3'
    h1_title: str = ""             # название раздела: 'Подписание договоров'
    h2_path: str = ""              # номер подраздела, если есть: '3.1'
    h2_title: str = ""
    level: int = 0
    table_index: int | None = None
    table_caption: str | None = None
    row_index: int | None = None
    header_row: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParsedDocument:
    """Результат разбора одного DOCX."""

    title: str
    meta: dict[str, str]
    blocks: list[ParsedBlock]
    source_path: str
    warnings: list[str] = field(default_factory=list)

    @property
    def doc_key(self) -> str:
        """Стабильный ключ семейства документов.

        Приоритет: явный doc_code из шапки -> stem файла без версии.

        Версию из имени файла срезаем обязательно: иначе
        reg_dogovora_v2.3.docx и reg_dogovora_v3.1.docx дадут разные doc_key,
        и фильтр актуальности (прямое решение боли «устаревшие регламенты»)
        просто не сработает — обе версии будут считаться разными документами.
        """
        code = self.meta.get("doc_code")
        if code:
            return code
        return _strip_version_from_stem(Path(self.source_path).stem)


# Хвост версии в имени файла: _v3.1, -v2, _3.1, _ver2.0, а также
# «_v2.3_final», «_v2.3_red», «_v2.3_итог» — версия может стоять не в конце.
# Хвостовые слова после версии разрешены явным списком, чтобы не съесть
# смысловые части имени («reg_dogovora_redakciya» не должно превратиться в «reg»).
_STEM_VERSION_RE = re.compile(
    r"[_\-\s]+(?:v|ver|версия|ред\.?)\s*\d+(?:[._]\d+)*(?:[_\-\s]*(?:final|fin|итог|red|редакция|new|new2|правка|ok|v\d+))?$"
    r"|[_\-\s]+\d+(?:[._]\d+)+$",
    re.IGNORECASE,
)


def _strip_version_from_stem(stem: str) -> str:
    """'reg_dogovora_v3.1' -> 'reg_dogovora'; 'prikaz_45_v2' -> 'prikaz_45'.

    Несколько проходов — на случай 'reg_v2.3_final' и подобных хвостов.
    """
    out = stem.strip()
    for _ in range(3):
        new = _STEM_VERSION_RE.sub("", out).strip("_- ")
        if new == out or not new:
            break
        out = new
    return out


def _style_name(p: Paragraph) -> str:
    try:
        return (p.style.name or "").strip()
    except Exception:  # noqa: BLE001 - повреждённые стили в рукописных DOCX
        return ""


def _cell_text(cell) -> str:
    """Текст ячейки в одну строку: переносы внутри ячейки схлопываются."""
    return " ".join(cell.text.split())


def _heading_level(style: str) -> int | None:
    """Level заголовка по имени стиля: Heading 1 / Заголовок 2."""
    low = style.lower()
    if low.startswith("heading"):
        tail = style[len("heading"):].strip().split(" ")[0]
        try:
            return int(tail)
        except ValueError:
            return 1
    if low.startswith("заголовок"):
        tail = style[len("заголовок"):].strip().split(" ")[0]
        try:
            return int(tail)
        except ValueError:
            return 1
    return None


def parse_docx(path: str | Path) -> ParsedDocument:
    """Разбирает DOCX в структурные блоки согласно контракту.

    Заголовки порождают section_path из собственной нумерации («1. Общие
    положения» -> '1'), иначе — из порядкового номера. Пункты без нумерации
    получают путь родительского раздела.
    """
    src = Path(path)
    doc = DocxDocument(str(src))

    blocks: list[ParsedBlock] = []
    warnings: list[str] = []
    title = ""
    meta: dict[str, str] = {}

    h1_n = h2_n = 0
    h1_path = h1_title = ""
    h2_path = h2_title = ""
    para_n = 0
    meta_consumed = False
    pending_caption: str | None = None
    table_n = 0

    # Порядок «абзац/таблица» восстанавливаем обходом тела документа:
    # doc.paragraphs и doc.tables раздельные и порядка не дают.
    para_iter = iter(doc.paragraphs)
    table_iter = iter(doc.tables)

    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]

        if tag == "p":
            try:
                p = next(para_iter)
            except StopIteration:
                break
            text = " ".join(p.text.split())
            if not text:
                continue
            style = _style_name(p)

            # 1) Заголовок документа
            if style.lower() == "title" and not title:
                title = text
                continue

            # 2) Шапка метаданных — один раз, сразу после Title
            if title and not meta_consumed and looks_like_meta_line(text):
                meta.update(parse_meta(text))
                meta_consumed = True
                continue

            # 3) Заголовки разделов
            level = _heading_level(style)
            if level is not None:
                num = ITEM_RE.match(text)
                if level <= 1:
                    h1_n += 1
                    h2_n = 0
                    h1_path = num.group(1) if num else str(h1_n)
                    h1_title = (num.group(2) if num else text).strip()
                    h2_path = h2_title = ""
                    blocks.append(ParsedBlock(
                        kind="h1", text=text, section_path=h1_path,
                        section_title=h1_title, level=1,
                    ))
                else:
                    h2_n += 1
                    h2_path = (num.group(1) if num
                               else f"{h1_path}.{h2_n}" if h1_path else str(h2_n))
                    h2_title = (num.group(2) if num else text).strip()
                    blocks.append(ParsedBlock(
                        kind="h2", text=text, section_path=h2_path,
                        section_title=h2_title or h1_title, level=2,
                    ))
                continue

            # 4) Подпись таблицы — прикрепляем к следующей таблице
            cap = TABLE_CAPTION_RE.match(text)
            if cap:
                pending_caption = text
                continue

            # 5) Нумерованный пункт
            num = ITEM_RE.match(text)
            if num and len(num.group(1)) <= 12:
                body = num.group(2).strip()
                if not body:
                    warnings.append(f"пустой пункт {num.group(1)}")
                    continue
                blocks.append(ParsedBlock(
                    kind="item", text=body, section_path=num.group(1),
                    section_title=h2_title or h1_title or title,
                    h1_path=h1_path, h1_title=h1_title,
                    h2_path=h2_path, h2_title=h2_title, level=3,
                ))
                continue

            # 6) Ненумерованный абзац: не теряем, это дословный источник цитат.
            para_n += 1
            sec = h2_path or h1_path
            item_path = f"{sec}.{para_n}" if sec else str(para_n)
            blocks.append(ParsedBlock(
                kind="item", text=text, section_path=item_path,
                section_title=h2_title or h1_title or title,
                h1_path=h1_path, h1_title=h1_title,
                h2_path=h2_path, h2_title=h2_title, level=3,
            ))

        elif tag == "tbl":
            try:
                table: Table = next(table_iter)
            except StopIteration:
                break
            table_n += 1
            rows = [[_cell_text(c) for c in row.cells] for row in table.rows]
            rows = [r for r in rows if any(c for c in r)]
            if not rows:
                continue
            header, body_rows = rows[0], rows[1:]
            caption, pending_caption = pending_caption, None

            for r_idx, row in enumerate(body_rows, start=1):
                # Построчный чанк: «Роль: Договор поставки; Сумма: до 100 000 руб.;
                # Подписант: Руководитель...» — читается человеком без таблицы.
                pairs = [f"{h}: {v}" for h, v in zip(header, row) if v]
                if not pairs:
                    continue
                blocks.append(ParsedBlock(
                    kind="table_row",
                    text="; ".join(pairs),
                    section_path=f"{h2_path or h1_path}#т{table_n}.{r_idx}",
                    section_title=caption or h2_title or h1_title or title,
                    h1_path=h1_path, h1_title=h1_title,
                    h2_path=h2_path, h2_title=h2_title, level=3,
                    table_index=table_n,
                    table_caption=caption,
                    row_index=r_idx,
                    header_row=header,
                ))

    if not title:
        title = src.stem
        warnings.append("не найден стиль Title — заголовком взято имя файла")

    # Связь версий может быть указана не в шапке, а в теле: «4.1. Настоящая
    # версия действует с 01.03.2026 и заменяет версию 2.3.» Для версионного
    # фильтра это критично: именно отсюда берётся valid_to предыдущей версии.
    if "replaces_version" not in meta:
        for b in blocks:
            m = _META_REPLACES.search(b.text)
            if m:
                meta["replaces_version"] = m.group(1)
                break
    # Дата вступления в силу тоже может быть только в теле («действует с ...»).
    if "valid_from" not in meta:
        for b in blocks:
            m = _META_VALID_FROM.search(b.text)
            if m:
                meta["valid_from"] = _iso(*m.groups())
                break

    if not meta:
        warnings.append("шапка метаданных не распознана")
    if not blocks:
        warnings.append("документ не содержит распознанных блоков")

    return ParsedDocument(
        title=title, meta=meta, blocks=blocks,
        source_path=str(src), warnings=warnings,
    )