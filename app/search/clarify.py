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

import json
import re
from pathlib import Path
from dataclasses import dataclass, field

from app.search.actors import actors_differ

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
    # Каскад: id записи реестра и путь выборов. Нужны, чтобы клиент
    # мог запросить следующий шаг у /api/clarify/step.
    clarify_id: str = ""
    path: list[int] = field(default_factory=list)


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

    # ВАЖНО: развилка — это РАЗНЫЕ СЦЕНАРИИ, а не просто близкие пункты.
    #
    # Ложное срабатывание: «за сколько ПА обязан принять заявку» давало
    # варианты «Сбор информации», «Сбор информации (п. 6.7)», «заявка,
    # находящаяся в статусе...» — три соседних раздела, ни один не
    # является сценарием. Пользователь не может выбрать: это не выбор,
    # а список глав.
    #
    # Признак настоящей развилки: у каждого варианта в тексте есть
    # УСЛОВИЕ («если», «в случае», «при наличии»). Именно так в
    # регламентах описаны ветви: «Если Заказчик выбрал продукт...».
    # Если условий нет — это не развилка, и уточнять нечего.
    distinct = [
        c for c in distinct
        if _extract_conditions(c.get("body") or "")
    ]
    if len(distinct) < MIN_OPTIONS:
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

    # Дедупликация меток: две кнопки с одинаковым текстом неразличимы.
    # Разные пункты одного раздела дают одинаковый section_title, поэтому
    # при совпадении добавляем номер пункта — он всегда разный.
    seen_labels: dict[str, int] = {}
    deduped: list[ClarificationOption] = []
    for o in options:
        if o.label in seen_labels:
            seen_labels[o.label] += 1
            suffix = f" (п. {o.section_path})" if o.section_path else ""
            o = ClarificationOption(
                label=o.label + suffix,
                section_path=o.section_path,
                doc_title=o.doc_title,
                score=o.score,
            )
        else:
            seen_labels[o.label] = 1
        if o.label not in {x.label for x in deduped}:
            deduped.append(o)
    options = deduped

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


# ---------------------------------------------------------------------------
# Междокументная неоднозначность (п. 7.5б брифинга)
#
# ДЕФЕКТ: на вопрос «кто назначает РП» система давала УВЕРЕННЫЙ ответ
# из одного регламента. По существу ответ зависит от этапа:
#   ТЗТ п. 4.1            — на пресейле: Руководитель отдела управления проектами
#   Проектная ДИБ п. 2.1  — в проекте: Координатор департамента
# Это два разных ответа на один вопрос. Уверенная цитата из одного
# документа скрывает второй — то есть пользователь получает неполный
# ответ, поданный как полный.
#
# Отличие от detect(): там развилка ВНУТРИ одного документа (выбор
# сценария), здесь — МЕЖДУ документами (выбор этапа или контура).
# ---------------------------------------------------------------------------

MIN_CROSS_DOC_SIM = 0.45

# Ключевые роли-исполнители для проверки согласия документов.
# Если два документа называют ОДНУ И ТУ ЖЕ ключевую роль, они
# согласны по существу, и развилка была бы навязанной. Своя уникальная
# роль у каждого — признак того, что ответ зависит от контура.
KEY_ROLES = {
    "РП", "ПА", "SS", "АМ", "TPS", "TSS", "ГИП", "АП", "РМ",
    "координатор департамента",
    "руководитель отдела УП", "руководитель группы УП",
    "руководитель департамента",
}

# Второе условие: пункт должен реально содержать слова вопроса.
# Без него детектор срабатывал на шуме: на «штраф за налоговую
# декларацию» косинусы тоже близки, но ответа в регламентах нет.
# Покрытие >= 0.5 означает «слова вопроса есть в тексте пункта».
MIN_CROSS_DOC_COVERAGE = 0.50


def detect_cross_document(
    candidates: list[dict],
    query: str,
    min_sim: float = MIN_CROSS_DOC_SIM,
    max_margin: float = MAX_MARGIN,
) -> Clarification | None:
    """Два регламента отвечают на один вопрос по-разному.

    Срабатывает, когда в топ-3 есть релевантные чанки из РАЗНЫХ документов
    с близкими косинусами: значит вопрос затрагивает несколько контуров,
    и выбрать один молча нельзя.
    """
    if len(candidates) < 2:
        return None

    # «Отвечающий» документ — тот, чей пункт содержит слова вопроса.
    # Косинус без покрытия даёт ложные срабатывания на шуме.
    relevant = [
        c for c in candidates
        if c.get("sim", 0) >= min_sim
        and c.get("coverage", 0.0) >= MIN_CROSS_DOC_COVERAGE
    ]
    if len(relevant) < 2:
        return None
    if len({c["doc_title"] for c in relevant}) < 2:
        return None

    # По одному лучшему чанку с каждого документа.
    best_per_doc: dict[str, dict] = {}
    for c in relevant:
        d = c["doc_title"]
        if d not in best_per_doc or c["sim"] > best_per_doc[d]["sim"]:
            best_per_doc[d] = c

    picked = sorted(best_per_doc.values(), key=lambda c: -c["sim"])[:MAX_OPTIONS]
    if len(picked) < 2:
        return None

    # ГЛАВНОЕ УСЛОВИЕ: документы должны называть РАЗНЫХ ИСПОЛНИТЕЛЕЙ.
    #
    # Косинус не различает «дополняют друг друга» и «называют разных
    # людей»: в обоих случаях он видит одну тему и близкие значения.
    # На «кто согласует ТКП» два документа называют одного и того же ПА —
    # развилка там была бы навязанной. На «кто назначает РП» один документ
    # называет Руководителя отдела УП, другой — Координатора департамента,
    # и показать только один значит подать неполный ответ как полный.
    #
    # Сравнение ролей детерминировано (app/search/actors.py): никакой
    # модели, только разбор текста пунктов.
    differ, roles = actors_differ(picked)
    if not differ:
        return None

    # Уточнение: если документы СХОДЯТСЯ хотя бы на одной ключевой роли,
    # они согласны по существу. «Кто согласует ТКП» — оба называют ПА,
    # значит ответ один. «Кто назначает РП» — один называет Руководителя
    # отдела УП, другой Координатора, общей роли нет, развилка нужна.
    key_sets = [set(r) & KEY_ROLES for r in roles.values()]
    key_sets = [k for k in key_sets if k]
    if len(key_sets) >= 2 and set.intersection(*key_sets):
        return None

    margin = picked[0]["sim"] - picked[-1]["sim"]

    options: list[ClarificationOption] = []
    for c in picked:
        doc_short = short_doc_name(c["doc_title"])
        section = (c.get("section_path") or "").strip()
        # Подпись: краткое имя документа + пункт. Читается за две секунды
        # и сразу показывает, ОТКУДА будет ответ.
        label = f"{doc_short}, п. {section}" if section else doc_short
        options.append(ClarificationOption(
            label=label, section_path=section,
            doc_title=c["doc_title"], score=round(float(c["sim"]), 4),
        ))

    if len(options) < MIN_OPTIONS:
        return None

    return Clarification(
        question=("На этот вопрос отвечают разные регламенты, и ответ зависит "
                  "от этапа или контура. Уточните, что имеется в виду:"),
        options=options,
        doc_title=", ".join(short_doc_name(c["doc_title"]) for c in picked),
        reason=f"разные документы, разрыв косинусов {margin:.3f}",
    )


# Краткие имена документов — только для кнопок.
# В текстах и источниках всегда полное название. Требование «без голого
# ДИБ» соблюдено: здесь «Пресейл ДИБ» и «Проектная ДИБ», их не спутать.
DOC_SHORT = {
    "Организация проектной деятельности": "ТЗТ",
    "Пресейловая деятельность Дирекции ИБ": "Пресейл ДИБ",
    "Проектная деятельность ДБРЦТ": "ДБРЦТ",
    "Проектная деятельность Дирекции ИБ": "Проектная ДИБ",
}


def short_doc_name(title: str) -> str:
    """Краткое имя документа для кнопки уточнения."""
    for key, short in DOC_SHORT.items():
        if title.startswith(key) or key.startswith(title):
            return short
    return title[:30]

# ---------------------------------------------------------------------------
# Явный реестр междокументных расхождений (config/conflicts.json).
#
# ПОЧЕМУ ЯВНЫЙ РЕЕСТР, А НЕ ЭВРИСТИКА. Автоматическое определение по
# косинусу и ролям дало ложные срабатывания в обе стороны: предлагало
# развилку на «кто согласует ТКП» (ответ однозначен) и не предлагало
# на «кто назначает РП» (расхождение реально). Косинус видит ТЕМУ,
# но не СОДЕРЖАНИЕ ответа.
#
# Междокументных расхождений в корпусе конечное число, и все они
# проверены по тексту. Реестр детерминирован, объясним и не ошибается
# молча: если расхождения нет в реестре, система не придумывает его.
# ---------------------------------------------------------------------------

_CONFLICT_CACHE: dict | None = None
_CONFLICT_PATH = Path(__file__).resolve().parents[2] / "config" / "conflicts.json"


def _load_conflicts() -> list[dict]:
    """Реестр расхождений из config/conflicts.json."""
    global _CONFLICT_CACHE
    if _CONFLICT_CACHE is None:
        try:
            _CONFLICT_CACHE = json.loads(
                _CONFLICT_PATH.read_text(encoding="utf-8")
            )
        except Exception:
            _CONFLICT_CACHE = {}
    return _CONFLICT_CACHE.get("conflicts", [])


def detect_registered_conflict(query: str) -> Clarification | None:
    """Расхождение из реестра, если вопрос про него.

    Триггеры сравниваются по вхождению в нормализованный запрос.
    Кнопки — СЕМАНТИЧЕСКИЕ («На этапе пресейла», «В проекте»),
    а не имена документов: для пользователя это выбор сценария,
    а не выбор регламента. Источник показывается внутри ветки.
    """
    norm = " ".join(query.lower().replace("ё", "е").split())
    for conflict in _load_conflicts():
        for trig in conflict.get("triggers", []):
            t = " ".join(trig.lower().replace("ё", "е").split())
            if t and t in norm:
                options = [
                    ClarificationOption(
                        label=o["label"],
                        section_path=o["section_path"],
                        doc_title=o["doc_title"],
                        score=1.0,
                    )
                    for o in conflict.get("options", [])
                ]
                return Clarification(
                    question=conflict["question"],
                    options=options,
                    doc_title=conflict.get("title", ""),
                    reason="известное расхождение регламентов",
                )
    return None

# ---------------------------------------------------------------------------
# Реестр развилок кейса 2 (config/clarifications.json).
#
# ПОЧЕМУ ЯВНЫЙ РЕЕСТР. Пороговая эвристика (разные section_path одного
# документа + близкие косинусы) давала ложные срабатывания: на «за сколько
# ПА обязан принять заявку» предлагала «Сбор информации», «Сбор информации
# (п. 6.7)» и «заявка в статусе Жду доп. инфо» — три соседние главы, ни одна
# не является сценарием. Выбирать между главами пользователь не может.
#
# Косинус видит близость темы, но не наличие ВЕТВЛЕНИЯ в тексте.
# Развилок в корпусе конечное число; все проверены дословно.
# ---------------------------------------------------------------------------

_CLAR_CACHE: dict | None = None
_CLAR_PATH = Path(__file__).resolve().parents[2] / "config" / "clarifications.json"


def _load_clarifications() -> list[dict]:
    """Реестр развилок из config/clarifications.json."""
    global _CLAR_CACHE
    if _CLAR_CACHE is None:
        try:
            _CLAR_CACHE = json.loads(
                _CLAR_PATH.read_text(encoding="utf-8")
            )
        except Exception:
            _CLAR_CACHE = {}
    return _CLAR_CACHE.get("clarifications", [])


def detect_registered_clarification(query: str) -> Clarification | None:
    """Развилка из реестра, если вопрос про неё.

    Возвращает первый шаг. Варианты — формулировки из регламента,
    а не придуманные: пользователь видит язык своего же документа.
    """
    norm = " ".join(query.lower().replace("ё", "е").split())
    for item in _load_clarifications():
        if item.get("internal"):
            continue
        for trig in item.get("triggers", []):
            t = " ".join(trig.lower().replace("ё", "е").split())
            if t and t in norm:
                steps = item.get("steps") or []
                if not steps:
                    continue
                step = steps[0]
                options = [
                    ClarificationOption(
                        label=o["label"],
                        section_path=o.get("next", ""),
                        doc_title=item.get("title", ""),
                        score=1.0,
                    )
                    for o in step.get("options", [])
                ]
                return Clarification(
                    question=step["question"],
                    options=options,
                    doc_title=item.get("title", ""),
                    reason="развилка из регламента: " + step.get("source", ""),
                    clarify_id=item.get("id", ""),
                    path=[],
                )
    return None

def resolve_registered_clarification(
    item_id: str, path: list[int]
) -> dict | None:
    """Разрешает каскад уточнений по выбранным вариантам.

    path — индексы выбранных вариантов на каждом шаге: [0] значит
    «выбрал первый вариант первого шага».

    Возвращает один из двух словарей:
      {'kind': 'step',    'question', 'options', 'source', 'step'}
      {'kind': 'answer',  'result', 'quote', 'doc_title', 'section_path'}

    Каскад нужен потому, что п. 1.3 Пресейл ДИБ устроен именно так:
    сначала «выбрал продукт или нет», затем «обозначил ли, что услуги
    не нужны», затем «≤30% или >30%». Один плоский вопрос здесь соврал бы.
    """
    for item in _load_clarifications():
        if item.get("id") != item_id:
            continue
        steps = item.get("steps") or []
        if not steps:
            return None

        step_idx = 0
        for choice in path or []:
            step = steps[step_idx]
            opts = step.get("options") or []
            if not (0 <= choice < len(opts)):
                return None
            chosen = opts[choice]
            if "next" in chosen:
                step_idx = int(chosen["next"])
                if step_idx >= len(steps):
                    return None
                continue
            # Дошли до результата: каскад завершён.
            return {
                "kind": "answer",
                "result": chosen.get("result", ""),
                "quote": chosen.get("quote", ""),
                "doc_title": chosen.get("doc_title", item.get("title", "")),
                "section_path": chosen.get("section_path", ""),
            }

        step = steps[step_idx]
        return {
            "kind": "step",
            "step": step_idx,
            "question": step["question"],
            "source": step.get("source", ""),
            "options": [o["label"] for o in step.get("options", [])],
            "clarify_id": item_id,
            "path": list(path or []),
        }
    return None

def conflict_branch_answer(
    conflict_id: str, label: str
) -> dict | None:
    """Цитата выбранной ветки междокументной развилки.

    Возвращает {result, quote, doc_title, section_path, also} — дословный
    текст пункта выбранного контура. Нужно, чтобы после ответа на развилку
    пользователь сразу видел цитату, а не уходил в повторный поиск.
    """
    for conflict in _load_conflicts():
        if conflict.get("id") != conflict_id:
            continue
        for opt in conflict.get("options", []):
            if opt.get("label") != label:
                continue
            return {
                "kind": "answer",
                "result": opt.get("note") or opt.get("quote", ""),
                "quote": opt.get("quote", ""),
                "doc_title": opt.get("doc_title", ""),
                "section_path": opt.get("section_path", ""),
                "also": opt.get("also") or [],
            }
    return None

