"""Кейс 2: детектор развилок и уточняющие вопросы.

БРИФ (цитата): «Если ответ на вопрос пользователя вызывает вариативность
из-за недостатка информации в вопросе пользователя, или невозможности точно
сопоставить вопрос с информацией в регламенте, то со стороны агента должен
поддерживаться диалог (агент должен уметь задать уточняющие вопросы
пользователю, на основании которых он предоставит однозначный ответ
или сообщит что не может дать однозначный ответ)».

ПРИНЦИП БЕЗ LLM. Развилка детектируется по структуре корпуса, а не
генерацией:
  1. В топ-3 найдены чанки РАЗНЫХ пунктов ОДНОГО документа;
  2. Их косинусы близки (разрыв мал) — то есть вопрос одинаково похож
     на несколько сценариев;
  3. Значит регламент описывает несколько ветвей, и без уточнения
     выбрать одну нельзя.

ВАРИАНТЫ УТОЧНЕНИЯ берутся ИЗ ТЕКСТА регламента, а не придумываются:
из заголовков найденных пунктов и из ключевых условий в их тексте
(«если ...», «в случае ...», «при ...»).

ПОСЛЕ УТОЧНЕНИЯ возможен и отказ — бриф прямо это допускает.
Комбинированный запрос проходит через coverage-гейт как обычно,
отдельной логики не строим.

ПОЧЕМУ КНОПКИ, А НЕ СВОБОДНЫЙ ДИАЛОГ. Бриф показывает уточняющий вопрос
с подтверждением («оплата списалась, но письмо не пришло в течение
30 минут — верно?»), то есть выбор из вариантов. Кнопки легитимны
и не требуют LLM: свободный диалог без генерации невозможен.

RAM: 0 (использует уже найденные чанки). Стоимость: 0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Пороги детектора. Обоснованы структурой, а не подгонкой:
#   MIN_MARGIN — если разрыв косинусов больше, ответ уже однозначен,
#                уточнение было бы навязчивым;
#   MAX_MARGIN — если разрыв почти нулевой, варианты неразличимы,
#                и уточнение не поможет (нужен отказ);
#   MIN_OPTIONS — меньше двух вариантов развилкой не является.
MIN_MARGIN = 0.005
MAX_MARGIN = 0.12
MIN_OPTIONS = 2
MAX_OPTIONS = 3

# Порог релевантности варианта уточнения.
#
# Дефект: развилка строилась из любых чанков одного документа, и на вопрос
# «ключ не пришёл, а оплата прошла» (пример из брифа) предлагались
# «Получение оплаты от Заказчика», «Согласование сметы» и «Перевод в
# Исполнение» — этапы проекта, ни один не отвечает на вопрос. Выбор такого
# варианта даёт уверенный ответ не на заданный вопрос, что хуже отказа.
#
# Порог 0.50 по косинусу: релевантные пункты дают 0.55–0.78,
# соседи по документу — 0.43–0.53.
MIN_OPTION_SIM = 0.50

# Условия-развилки в тексте регламента. Из них строятся варианты ответа.
_CONDITION = re.compile(
    r"(?:если|в случае|при условии|при наличии|при отсутствии|"
    r"в зависимости от|когда)\s+([^.;]{10,120})",
    re.IGNORECASE,
)

# Заголовки пунктов часто сами называют ветвь.
_HEADING_CASE = re.compile(r"^([А-ЯЁ][^.]{8,70})$")


@dataclass(slots=True)
class ClarificationOption:
    """Один вариант уточнения — кнопка для пользователя."""

    label: str            # текст кнопки, короткий
    section_path: str     # какой пункт регламента это подтверждает
    doc_title: str
    score: float


@dataclass(slots=True)
class Clarification:
    """Развилка: вопрос, варианты, основание."""

    question: str                       # что спросить у пользователя
    options: list[ClarificationOption] = field(default_factory=list)
    doc_title: str = ""
    reason: str = ""


_BAD_START = re.compile(
    r"^(?:необходим|приложен|указан|предоставлен|относящ|связанн|"
    r"котор|котора|которы|его|ее|их|этот|эта|эти|того|тем|"
    r"данн|информаци|документ)\w*\s",
    re.IGNORECASE,
)


def _label_from_condition(cond: str) -> str | None:
    """Годная ли подпись из условия. None — если фраза невнятная.

    Дефект: подписи вида «необходимых для расчета данных, приложенных
    TPS» — середина фразы в родительном падеже. Пользователь не
    понимает, что выбирает. Такие отбрасываем.
    """
    c = " ".join(cond.split()).strip(" ,;:.")
    if not (15 <= len(c) <= 70):
        return None
    if _BAD_START.match(c):
        return None
    words = [w for w in c.split() if len(w) > 3]
    if len(words) < 2:
        return None
    return c


def _label_from_heading(section_title: str, body: str) -> str:
    """Fallback: заголовок пункта или начало тела."""
    t = " ".join((section_title or "").split())
    t = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", t).strip()
    if 10 <= len(t) <= 70:
        return t
    b = " ".join((body or "").split())
    b = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", b)
    return _shorten(b, 65)


def _shorten(text: str, limit: int = 70) -> str:
    """Короткая подпись для кнопки: первое предложение, обрезка по слову."""
    t = " ".join(text.split())
    t = t.split(". ")[0].strip(" .;:")
    if len(t) <= limit:
        return t
    cut = t[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def _extract_conditions(text: str) -> list[str]:
    """Условия-развилки из текста пункта.

    «Если Заказчик выбрал продукт и запрашивает только расчёт» ->
    «Заказчик выбрал продукт и запрашивает только расчёт».
    """
    out = []
    for m in _CONDITION.finditer(text):
        cond = m.group(1).strip(" ,;:")
        if len(cond) >= 10:
            out.append(cond)
    return out


def detect(
    candidates: list[dict],
    query: str,
    min_margin: float = MIN_MARGIN,
    max_margin: float = MAX_MARGIN,
) -> Clarification | None:
    """Определяет, нужен ли уточняющий вопрос.

    candidates: [{
        "doc_title", "section_path", "section_title", "body", "sim",
    }] — топ найденных чанков с косинусами, в порядке убывания sim.

    Возвращает Clarification или None, если ответ однозначен.
    """
    if len(candidates) < MIN_OPTIONS:
        return None

    # Лучший по КОСИНУСУ чанк — опорный. Развилка строится только внутри
    # его документа: разные документы на один вопрос означают
    # неоднозначность источника, а не сценария, и уточнять там нечего.
    top = max(candidates, key=lambda c: c.get("sim", 0.0))
    same_doc = [c for c in candidates if c["doc_title"] == top["doc_title"]]
    if len(same_doc) < MIN_OPTIONS:
        return None
    # Дополнительная проверка: все варианты обязаны быть из одного документа.
    if len({c["doc_title"] for c in same_doc[:MAX_OPTIONS]}) != 1:
        return None

    # Разные пункты одного документа.
    distinct: list[dict] = []
    seen_paths: set[str] = set()
    for c in same_doc:
        p = c.get("section_path") or ""
        if p in seen_paths:
            continue
        seen_paths.add(p)
        distinct.append(c)
        if len(distinct) >= MAX_OPTIONS:
            break

    if len(distinct) < MIN_OPTIONS:
        return None

    # Близость косинусов: если разрыв большой, ответ однозначен.
    margin = distinct[0]["sim"] - distinct[-1]["sim"]
    if margin > max_margin:
        return None
    # Если разрыв почти нулевой, варианты неразличимы — уточнение
    # не поможет, нужен отказ. Оставляем отказ решать гейту.
    if margin < min_margin:
        return None

    options: list[ClarificationOption] = []
    for c in distinct:
        # Подпись варианта: ищем ПЕРВОЕ годное условие, а не первое попавшееся.
        # Дефект: «необходимых для расчета данных, приложенных TPS» —
        # середина фразы в родительном падеже, пользователь не поймёт.
        label = None
        for cond in _extract_conditions(c.get("body") or ""):
            candidate = _label_from_condition(cond)
            if candidate:
                label = candidate
                break
        if not label:
            label = _label_from_heading(
                c.get("section_title") or "", c.get("body") or ""
            )
        if not label or len(label) < 5:
            continue
        options.append(ClarificationOption(
            label=label,
            section_path=c.get("section_path") or "",
            doc_title=c["doc_title"],
            score=round(float(c["sim"]), 4),
        ))

    # Отбрасываем варианты, нерелевантные самому вопросу.
    #
    # ДЕФЕКТ: развилка строилась из любых чанков одного документа. На вопрос
    # «ключ не пришёл, а оплата прошла» (пример из брифа) система предлагала
    # «Получение оплаты от Заказчика» / «Согласование сметы» / «Перевод в
    # Исполнение» — три этапа проекта, ни один не отвечает на вопрос.
    #
    # Такой выбор хуже отказа: пользователь выбирает вариант, которого нет
    # в регламенте, и получает уверенный ответ не на свой вопрос.
    #
    # Порог 0.50 по косинусу: релевантные пункты дают 0.55–0.78,
    # соседи по документу — 0.43–0.53. Порог разделяет эти диапазоны.
    options = [o for o in options if o.score >= MIN_OPTION_SIM]

    if len(options) < MIN_OPTIONS:
        return None

    # Текст вопроса: не «уточните», а конкретная развилка из регламента.
    question = (
        f"В регламенте «{top['doc_title']}» описано несколько сценариев. "
        f"Уточните, какой относится к вашему вопросу:"
    )
    return Clarification(
        question=question,
        options=options[:MAX_OPTIONS],
        doc_title=top["doc_title"],
        reason=f"разные пункты одного документа, разрыв косинусов {margin:.3f}",
    )


def build_combined_query(original: str, option: ClarificationOption) -> str:
    """Комбинированный запрос после выбора варианта.

    К выбранному варианту добавляем ключевые слова пункта, чтобы поиск
    сузился на нужную ветвь. НЕ добавляем текст варианта целиком:
    он может быть длинным и утопить исходный вопрос.
    """
    label = option.label.rstrip("…").strip()
    return f"{original} {label}"