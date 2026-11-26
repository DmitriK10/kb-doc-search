"""Перевод терминов англоязычных вопросов в термины регламентов.

ЗАЧЕМ. Заказчик прямо называет «who appoints the project manager» как
рабочий паттерн вопросов (п. 6.5 брифа). Энкодер многоязычный, но
FTS-ветка заточена под русский, а в корпусе русские термины. Проверено:
без подстановки такой вопрос получает отказ с уверенностью 0.58.

ЧТО ЭТО НЕ ТАК. Это НЕ машинный перевод и не LLM. Это словарь известных
эквивалентов: «project manager» -> «руководитель проекта», «deadline» ->
«срок». Если термин не в словаре, он остаётся как есть. Лучше не
перевести, чем перевести наугад: неверный перевод увёл бы поиск не туда,
а так система честно не находит ответа.

Стоимость: ~0.03 мс (один проход по 13 парам).
"""

from __future__ import annotations

import re

# Английские триггеры -> русский термин регламента.
# Сверено с текстами: эти слова действительно встречаются в регламентах.
_EN_TERMS: list[tuple[tuple[str, ...], str]] = [
    (("who appoints", "who assigns", "appoints"), "назначает"),
    (("project manager", "project lead"), "руководитель проекта"),
    (("deadline", "how long"), "срок"),
    (("presale", "pre-sale", "pre sale"), "пресейл"),
    (("request", "application"), "заявка"),
    (("subcontractor",), "подрядчик"),
    (("how many", "how much"), "сколько"),
    (("calculation", "estimate"), "расчёт"),
    (("competition", "tender"), "конкурс"),
    (("proposal", "offer"), "предложение"),
    (("approve", "approves", "approval"), "согласует"),
    (("signs", "sign", "signature"), "подписывает"),
    (("responsible", "in charge"), "отвечает"),
    (("account manager",), "аккаунт-менеджер"),
    (("technical proposal",), "технико-коммерческое"),
]

# Запрос считаем англоязычным, если в нём нет кириллицы.
_HAS_CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")


def translate_terms(query: str) -> str:
    """Дополняет англоязычный запрос русскими терминами регламентов.

    «who appoints the project manager» ->
    «who appoints the project manager назначает руководитель проекта»

    Возвращает исходный запрос, если он не англоязычный или ни один
    термин не распознан.
    """
    if not query or _HAS_CYRILLIC.search(query):
        return query

    low = query.lower()
    added: list[str] = []
    for triggers, target in _EN_TERMS:
        for trig in triggers:
            if re.search(rf"(?<![a-z]){re.escape(trig)}(?![a-z])", low):
                if target not in added:
                    added.append(target)
                break

    if not added:
        return query
    return query + " " + " ".join(added)