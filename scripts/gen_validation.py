"""Генератор валидационного набора: вопросы + эталонные чанки.

Зачем отдельный генератор, а не «напишем 150 вопросов руками»: вручную
за 4 часа получается набор из копий формулировок из документов, и он
не проверяет главное — умение находить ответ, когда спрашивают ДРУГИМИ
словами. Здесь для каждого пункта строится несколько регистров:
  - прямой: близко к тексту документа;
  - бытовой: как спрашивает сотрудник, не читавший регламент;
  - с синонимом/опечаткой: проверка устойчивости.

Разметка:
  in-scope    — вопрос имеет ответ в корпусе, gold_chunk_ids заполнены;
  out-of-scope — темы, которых в базе нет (доверенности, зарубежные
                 командировки, ИБ-инциденты). Это 20-25% реального потока,
                 и именно на них система обязана отказывать.

Сплит dev/test фиксирован seed'ом: порог подбирается на dev, цифры на
слайде — только с test. Иначе получается подгонка под демо.

Запуск: python scripts/gen_validation.py --corpus data/corpus
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest.chunker import chunk_document  # noqa: E402
from app.ingest.docx_parser import parse_docx  # noqa: E402

# Шаблоны бытовых вопросов по типу пункта. Ключ — «как звучит» тема пункта.
# Это не универсальный генератор: список подобран под темы корпуса и
# расширяется по мере роста корпуса.
TEMPLATES = {
    "подписание": [
        "кто подписывает {obj}",
        "кто должен подписать {obj}",
        "{obj} — кто подписант",
        "чей подписью оформляется {obj}",
    ],
    "срок": [
        "в какой срок {act}",
        "сколько времени даётся на {act}",
        "какой срок {act}",
        "не позднее какого срока {act}",
    ],
    "порядок": [
        "как {act}",
        "каким образом {act}",
        "что нужно чтобы {act}",
        "какой порядок {act}",
    ],
    "сумма": [
        "какая сумма {obj}",
        "сколько составляет {obj}",
        "в каком размере {obj}",
    ],
    "ответственный": [
        "кто отвечает за {obj}",
        "кто ответственный за {obj}",
        "на кого возложен {obj}",
    ],
}

# Бытовые переформулировки: как сотрудник спросит, не зная терминов.
COLLOQUIAL = {
    "Срок согласования — не позднее 5 рабочих дней с момента передачи проекта "
    "договора ответственному исполнителю.":
        "сколько ждать согласования договора",
    "Агентский договор подписывает финансовый директор при сумме свыше "
    "1 000 000 рублей.":
        "кто подписывает агентский договор свыше миллиона",
    "Суточные выплачиваются в размере 700 рублей за каждый день нахождения "
    "в командировке.":
        "размер суточных в командировке",
    "Отчёт о командировке сдаётся в бухгалтерию не позднее 3 рабочих дней "
    "после возвращения.":
        "когда сдавать отчёт после командировки",
    "Заявление на отпуск подаётся не позднее чем за 14 календарных дней "
    "до начала.":
        "за сколько дней писать заявление на отпуск",
    "Минимальная длина пароля — 12 символов, смена не реже одного раза "
    "в 90 дней.":
        "какие требования к паролю",
    "Договоры на сумму до 100 000 рублей подписывает руководитель "
    "подразделения инициатора.":
        "кто подписывает мелкие договоры до ста тысяч",
    "Закупки на сумму свыше 3 000 000 рублей проводятся открытым тендером.":
        "с какой суммы нужен тендер на закупку",
    "Ежегодный основной оплачиваемый отпуск — 28 календарных дней.":
        "сколько дней отпуска в год",
    "Договоры хранятся 5 лет после истечения срока действия.":
        "сколько хранятся договоры в архиве",
    "Критические инциденты (недоступность системы) решаются в течение "
    "2 часов.":
        "как быстро чинят критичные сбои в ИТ",
    "Обучение стоимостью свыше 50 000 рублей требует согласования "
    "финансового директора.":
        "кто согласует оплату обучения дороже пятидесяти тысяч",
}


def classify(item_text: str) -> str:
    """Определяет тип пункта по ключевым словам — для выбора шаблона."""
    low = item_text.lower()
    if any(w in low for w in ("подписыва", "подписант", "визир")):
        return "подписание"
    if any(w in low for w in ("срок", "не позднее", "в течение", "дней")):
        return "срок"
    if any(w in low for w in ("рублей", "процент", "размере", "сумм")):
        return "сумма"
    if any(w in low for w in ("ответствен", "отвечает", "возложен")):
        return "ответственный"
    return "порядок"


def main() -> int:
    ap = argparse.ArgumentParser(description="Генерация валидационного набора")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    corpus = Path(args.corpus)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    doc_type_by_file = {d["file"]: d for d in manifest["documents"]}

    # Собираем пункты с их чанками. Используем ТОТ ЖЕ чанкер, что индексатор:
    # иначе gold_chunk_ids не совпадут с тем, что лежит в БД.
    pool: list[dict] = []
    for path in sorted(corpus.glob("*.docx")):
        doc = parse_docx(path)
        chunks = chunk_document(doc)
        info = doc_type_by_file.get(path.name, {})
        # Из недействительных версий вопросы не строим: спрашивать про
        # отменённый регламент — отдельный сценарий, он проверяется
        # вручную и не должен попадать в Recall@3.
        if info.get("valid_to"):
            continue
        for ch in chunks:
            if ch.char_len < 60 or ch.chunk_type != "item":
                continue
            pool.append({
                "file": path.name,
                "doc_title": doc.title,
                "version": doc.meta.get("version"),
                "section_path": ch.section_path,
                "section_title": ch.section_title,
                "body": ch.body,
                "char_len": ch.char_len,
            })

    print(f"пунктов-кандидатов: {len(pool)}")

    # --- in-scope: по одному вопросу на пункт, чередуя регистры ---
    records: list[dict] = []
    seen_questions: set[str] = set()

    for item in pool:
        body = item["body"]
        question = None

        # 1) Бытовой вопрос, если он есть в словаре — самый ценный:
        #    проверяет поиск по смыслу, а не по совпадению слов.
        question = COLLOQUIAL.get(body)

        # 2) Иначе — по шаблону, с извлечением объекта из текста.
        if question is None:
            kind = classify(body)
            tmpl = rnd.choice(TEMPLATES[kind])
            obj = _extract_object(body)
            act = _extract_action(body)
            try:
                question = tmpl.format(obj=obj, act=act)
            except KeyError:
                question = None

        if not question:
            continue
        q_norm = question.lower().strip()
        if q_norm in seen_questions:
            continue
        seen_questions.add(q_norm)

        records.append({
            "query": question,
            "gold": {
                "file": item["file"],
                "doc_title": item["doc_title"],
                "version": item["version"],
                "section_path": item["section_path"],
            },
            "out_of_scope": False,
            "register": "colloquial" if question == COLLOQUIAL.get(body) else "template",
        })

    # --- out-of-scope: темы, которых нет в базе ---
    for topic in manifest["empty_topics"]:
        for q in topic["sample_questions"]:
            records.append({
                "query": q,
                "gold": None,
                "out_of_scope": True,
                "topic": topic["topic_key"],
                "expected_owner_role": topic["owner_role"],
                "register": "out_of_scope",
            })

    # --- Сплит dev/test фиксированный ---
    rnd.shuffle(records)
    in_scope = [r for r in records if not r["out_of_scope"]]
    out_scope = [r for r in records if r["out_of_scope"]]

    n_in_dev = int(len(in_scope) * 0.6)
    dev = in_scope[:n_in_dev] + out_scope[: int(len(out_scope) * 0.6)]
    test = in_scope[n_in_dev:] + out_scope[int(len(out_scope) * 0.6):]
    rnd.shuffle(dev)
    rnd.shuffle(test)

    for name, split in (("validation_dev.jsonl", dev), ("validation_test.jsonl", test)):
        path = out_dir / name
        with path.open("w", encoding="utf-8") as fh:
            for rec in split:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_pos = sum(1 for r in split if not r["out_of_scope"])
        n_neg = sum(1 for r in split if r["out_of_scope"])
        print(f"{name:26s} всего={len(split):3d} in-scope={n_pos:3d} out-of-scope={n_neg:3d}")

    print(f"\nвсего записей: {len(records)}")
    return 0


def _extract_object(body: str) -> str:
    """Грубое извлечение объекта из пункта для подстановки в шаблон.

    Не претендует на синтаксический разбор: нужно получить короткую
    фразу, по которой человек поймёт вопрос. Если не получилось —
    возвращаем «этот вопрос», и шаблон даёт общую формулировку.
    """
    low = body.lower()
    for marker, tail in (
        ("подписывает ", ""), ("срок ", ""), ("в размере ", ""),
        ("на сумму ", ""), ("за ", ""),
    ):
        idx = low.find(marker)
        if idx >= 0:
            frag = body[idx + len(marker):].strip()
            frag = frag.split(".")[0].split(",")[0]
            if 3 < len(frag) < 60:
                return frag.lower()
    return "этот вопрос"


def _extract_action(body: str) -> str:
    """Извлекает действие для шаблонов «как ...», «в какой срок ...»."""
    low = body.lower()
    for marker in ("оформляется ", "проходит ", "подаётся ", "сдаётся ",
                   "проводится ", "выплачиваются ", "согласуется "):
        idx = low.find(marker)
        if idx >= 0:
            frag = body[idx + len(marker):].strip().split(".")[0]
            if 3 < len(frag) < 70:
                return frag.lower()
    return "оформить это"


if __name__ == "__main__":
    raise SystemExit(main())