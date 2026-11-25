"""Генератор демонстрационной истории запросов.

Зачем это нужно и почему это честно.

Проблема: при потоке 10 запросов в день за неделю набирается ~17 отказов.
KMeans на 17 точках даёт кластеры по 5 штук, центроиды нестабильны, и
«красная зона» на дашборде либо не появляется, либо прыгает между
прогонами. Аналитика, которая работает раз в месяц, для демо бесполезна.

Решение: 30-дневная история прогоняется через НАСТОЯЩИЙ пайплайн поиска.
В query_logs попадают реальные эмбеддинги, реальные отказы, реальные
латентности. Синтетическими являются только ФОРМУЛИРОВКИ вопросов и
таймстемпы.

Что честно помечается:
  is_simulated = TRUE у всех сгенерированных строк. KPI на дашборде
  считается ТОЛЬКО по is_simulated = FALSE, синтетика показывается
  отдельной панелью с явным заголовком. Смешивать измерение и
  демонстрацию нельзя — иначе цифра на слайде становится фикцией.

Таймстемпы распределены по суточному профилю: рабочее время 9:00-19:00,
85% будни, 15% выходные. Без этого Grafana показала бы всплеск всех
запросов в один час, и подделку было бы видно по графику.

Запуск:
    python scripts/gen_query_logs.py --dsn "..." --days 30 --per-day 13
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.search.encoder import Encoder, to_pgvector  # noqa: E402
from app.search.search import Searcher  # noqa: E402

# Формулировки вопросов: 70% по темам, которые ЕСТЬ в базе (успешные ответы),
# 22% по темам, которых НЕТ (отказы), 8% про устаревшие версии.
# Пропорция 22% отказов — из брифа: «до 40% запросов остаются без ответа»,
# мы моделируем консервативный сценарий после внедрения.
IN_SCOPE = [
    "кто подписывает агентский договор свыше миллиона",
    "кто подписывает договоры до ста тысяч рублей",
    "какой срок согласования договора",
    "сколько ждать согласования договора",
    "кто подписывает договор от ста тысяч до миллиона",
    "кто подписывает договоры от имени компании",
    "что делать при разногласиях по проекту договора",
    "у кого согласовывается проект договора",
    "размер суточных в командировке",
    "за сколько дней оформляется командировка",
    "когда сдавать отчёт после командировки",
    "когда возмещают расходы на проезд",
    "сколько платят суточных за день",
    "с какой суммы закупки проводят через тендер",
    "сколько поставщиков нужно для запроса предложений",
    "до какой суммы можно купить у единственного поставщика",
    "сколько рассматривают заявку на закупку",
    "кто согласует заявку на закупку",
    "сколько действует соглашение о конфиденциальности",
    "кто готовит проект NDA",
    "можно ли менять типовой шаблон NDA",
    "какие требования к паролю",
    "можно ли пользоваться личной флешкой",
    "как часто менять пароль",
    "за сколько дней писать заявление на отпуск",
    "сколько дней отпуска в год",
    "когда утверждают график отпусков",
    "сколько хранятся договоры в архиве",
    "сколько лет хранят кадровые документы",
    "как быстро зафиксировать контакт с клиентом в CRM",
    "какие поля обязательны в CRM",
    "при каком выполнении плана дают квартальную премию",
    "положена ли премия если отработал меньше трёх месяцев",
    "сколько дней в неделю можно работать удалённо",
    "кто согласует полную удалённую работу",
    "кто согласует оплату обучения дороже пятидесяти тысяч",
    "можно ли ставить печать на пустом бланке",
    "где хранятся печати компании",
    "как быстро чинят критичные сбои",
    "сколько ждать решения обычной заявки",
    "как часто проходят медосмотр",
    "кто платит за медосмотр",
    "что делать при возгорании",
    "можно ли пользоваться лифтом при эвакуации",
    "когда регистрируют входящую корреспонденцию",
    "срок исполнения срочных документов",
    "в каких пределах возмещают ущерб",
    "что грозит за разглашение коммерческой тайны",
]

OUT_OF_SCOPE = [
    "как получить доверенность на подписание договоров",
    "кто выписывает доверенность на представление интересов",
    "нужна ли доверенность сотруднику для подписания акта",
    "сколько действует доверенность на сделку",
    "как отозвать выданную доверенность",
    "как оформить командировку за границу",
    "какие документы нужны для выезда за рубеж",
    "кто оплачивает визу для зарубежной командировки",
    "как получить аванс в валюте на загранкомандировку",
    "что делать при утечке данных",
    "как сообщить об инциденте информационной безопасности",
    "кто расследует инцидент ИБ",
    "в какой срок уведомить об утечке персональных данных",
    "как получить пропуск в офис для подрядчика",
    "кто согласует аренду переговорной",
    "как заказать корпоративную сим-карту",
    "размер компенсации за личный автомобиль",
    "как перевести сотрудника в другой отдел",
    "как оформить больничный",
    "кто подписывает справку о доходах",
]

STALE = [
    "какая версия регламента договоров действует",
    "что изменилось в регламенте договоров",
    "действует ли ещё регламент версии 2.3",
]

USERS = [f"user{i:02d}" for i in range(1, 15)]


def make_timestamp(rnd: random.Random, day: datetime) -> datetime:
    """Время внутри дня по суточному профилю.

    Рабочие часы 9-19 с пиком 10-12 и 15-17. Выходные — редкие запросы.
    Без профиля Grafana покажет ровный шум, что выглядит неестественно.
    """
    weekday = day.weekday()  # 0=Пн
    if weekday >= 5 and rnd.random() > 0.15:
        return None  # в выходные 85% дней без запросов

    # Пики: утро и после обеда
    if rnd.random() < 0.40:
        hour = rnd.randint(10, 12)
    elif rnd.random() < 0.70:
        hour = rnd.randint(14, 17)
    else:
        hour = rnd.randint(9, 18)
    minute = rnd.randint(0, 59)
    second = rnd.randint(0, 59)
    return day.replace(hour=hour, minute=minute, second=second)


def main() -> int:
    ap = argparse.ArgumentParser(description="Генерация демо-истории запросов")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--per-day", type=int, default=13)
    ap.add_argument("--model-dir", default="/opt/kb/models/rubert-tiny2-int8")
    ap.add_argument("--config", default="config/refusal.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--clean", action="store_true",
                    help="удалить прежнюю синтетическую историю")
    args = ap.parse_args()

    import psycopg

    rnd = random.Random(args.seed)
    enc = Encoder(args.model_dir)
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    searcher = Searcher(args.dsn, enc, "demo-pepper", cfg)

    if args.clean:
        with psycopg.connect(args.dsn) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM query_logs WHERE is_simulated = TRUE")
            conn.commit()
            print(f"удалено прежних синтетических: {cur.rowcount}")
            cur.close()

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=args.days)
    total = 0
    rows: list[dict] = []

    print(f"генерация {args.days} дней × ~{args.per_day} запросов...")
    for d in range(args.days):
        day = start + timedelta(days=d)
        ts = make_timestamp(rnd, day)
        if ts is None:
            continue
        n = max(1, int(rnd.gauss(args.per_day, 3)))

        for _ in range(n):
            roll = rnd.random()
            if roll < 0.22:
                q = rnd.choice(OUT_OF_SCOPE)
            elif roll < 0.28:
                q = rnd.choice(STALE)
            else:
                q = rnd.choice(IN_SCOPE)

            t = ts.replace(hour=min(23, max(0, ts.hour + rnd.randint(-1, 1))),
                           minute=rnd.randint(0, 59))
            rows.append({"query": q, "ts": t, "user": rnd.choice(USERS)})

    rows.sort(key=lambda r: r["ts"])
    print(f"сгенерировано записей: {len(rows)}")

    # Прогон через НАСТОЯЩИЙ поиск: эмбеддинги и решения реальные.
    with psycopg.connect(args.dsn, autocommit=False) as conn:
        cur = conn.cursor()
        for i, r in enumerate(rows, 1):
            ans = searcher.search(r["query"], user_id=r["user"], use_cache=False)
            qvec = enc.encode_one(r["query"])
            qhmac = hashlib.sha256(
                f"{r['query']}|{r['ts'].isoformat()}".encode()
            ).hexdigest()

            # Обратная связь: 12% пользователей ставят оценку.
            # Успешные ответы чаще «полезно», отказы чаще «бесполезно» —
            # так выглядит реальное распределение.
            fb = None
            if rnd.random() < 0.12:
                if ans.response_type == "refusal":
                    fb = 1 if rnd.random() < 0.25 else -1
                else:
                    fb = 1 if rnd.random() < 0.82 else -1

            cur.execute("""
                INSERT INTO query_logs
                    (created_at, query_hmac, query_len, embedding, response_type,
                     refusal_reason, confidence, top_score, sources, latency_ms,
                     cache_hit, user_id_hash, user_feedback, feedback_at,
                     is_simulated)
                VALUES (%s, %s, %s, %s::halfvec, %s, %s, %s, %s, %s::jsonb,
                        %s, false, %s, %s, %s, true)
            """, (
                r["ts"], qhmac, len(r["query"]), to_pgvector(qvec),
                ans.response_type, ans.refusal_reason, ans.confidence,
                ans.features.get("max_sentence_sim"),
                json.dumps([s.to_dict() for s in ans.sources], ensure_ascii=False),
                ans.latency_ms,
                hashlib.sha256(r["user"].encode()).hexdigest(),
                fb, r["ts"] if fb is not None else None,
            ))
            total += 1
            if i % 50 == 0:
                conn.commit()
                print(f"  {i}/{len(rows)}")

        conn.commit()
        cur.execute("""
            SELECT count(*),
                   count(*) FILTER (WHERE response_type = 'success'),
                   count(*) FILTER (WHERE response_type = 'stale'),
                   count(*) FILTER (WHERE response_type = 'refusal'),
                   count(*) FILTER (WHERE user_feedback IS NOT NULL)
            FROM query_logs WHERE is_simulated = TRUE
        """)
        t, s, st, rf, fbn = cur.fetchone()
        cur.close()

    print(f"\nвсего: {t}")
    print(f"  success: {s} ({s/t*100:.1f}%)")
    print(f"  stale:   {st} ({st/t*100:.1f}%)")
    print(f"  refusal: {rf} ({rf/t*100:.1f}%)")
    print(f"  с оценкой: {fbn}")
    print("\nВСЕ строки помечены is_simulated = TRUE.")
    print("KPI на дашборде считается по is_simulated = FALSE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())