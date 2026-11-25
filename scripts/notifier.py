"""Транспорт алертов: pending → отправлено владельцу.

Зачем отдельный модуль. Логика алертов живёт в PostgreSQL (`knowledge_alerts`),
и это правильно: транспорт заменяем. Здесь только доставка.

Два режима:
  1. Telegram — если задан TELEGRAM_TOKEN и у владельца есть chat_id.
  2. Dry-run — если токена нет: помечает алерты как sent и печатает текст.
     Это НЕ заглушка в кадре: в демо показывается лента алертов на стенде,
     а notifier в dry-run нужен для проверки логики без реального бота.

Почему не n8n: n8n распространяется по Sustainable Use License (не OSI),
и его нет в текущей поставке. Логика в БД, транспорт — этот скрипт.

Дедупликация уже в схеме: UNIQUE (cluster_id, reason, created_day).
Скрипт дополнительно проверяет status='pending', чтобы не спамить.

Запуск:
    python scripts/notifier.py --dsn "..." --dry-run
    python scripts/notifier.py --dsn "..." --token "$TELEGRAM_TOKEN"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


def format_alert(payload: dict) -> str:
    """Текст уведомления владельцу.

    Формулировка задачи, а не пересказ метрик: что случилось, что делать,
    куда смотреть. Владелец должен понять за 5 секунд.
    """
    zone = payload.get("zone", "red")
    topic = payload.get("topic") or "тема без названия"
    size = payload.get("size", 0)
    refusals = payload.get("refusal_share", 0)
    useless = payload.get("useless_share", 0)

    if zone == "red":
        head = f"Нет ответа в базе знаний: «{topic}»"
        detail = (f"За период {size} запросов, из них {refusals:.0%} "
                  f"получили отказ.")
    elif zone == "yellow":
        head = f"Ответ есть, но не помогает: «{topic}»"
        detail = (f"За период {size} запросов, {useless:.0%} помечены "
                  f"как бесполезные.")
    else:
        head = f"Тема в норме: «{topic}»"
        detail = f"{size} запросов, {refusals:.0%} отказов."

    parts = [
        f"⚠ {head}",
        "",
        detail,
        "",
        "Что сделать:",
        "1. Проверить, есть ли действующий регламент по теме.",
        "2. Если есть — загрузить в базу знаний.",
        "3. Если нет — создать документ или дать ссылку на существующий.",
    ]
    if payload.get("nearest_doc"):
        parts.append(f"\nБлижайший документ в базе: {payload['nearest_doc']}")
    if payload.get("owner_role"):
        parts.append(f"Владелец темы: {payload['owner_role']}")
    return "\n".join(parts)


def send_telegram(token: str, chat_id: str, text: str) -> bool:
    """Отправка через Telegram Bot API. Возвращает успех."""
    if httpx is None:
        print("httpx не установлен", file=sys.stderr)
        return False
    try:
        r = httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10.0,
        )
        if r.status_code >= 300:
            print(f"  Telegram ответил {r.status_code}: {r.text[:150]}",
                  file=sys.stderr)
            return False
        return True
    except httpx.HTTPError as exc:
        print(f"  ошибка сети: {exc}", file=sys.stderr)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Отправка алертов владельцам")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--token", default="", help="Telegram bot token")
    ap.add_argument("--chat-id", default="", help="chat_id получателя")
    ap.add_argument("--dry-run", action="store_true",
                    help="не отправлять, только пометить sent и напечатать")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    import psycopg

    with psycopg.connect(args.dsn, autocommit=False) as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, cluster_id, reason, payload, created_day
            FROM knowledge_alerts
            WHERE status = 'pending'
            ORDER BY created_day DESC, id
            LIMIT %s
        """, (args.limit,))
        rows = cur.fetchall()

        if not rows:
            print("нет алертов со статусом pending")
            cur.close()
            return 0

        print(f"алертов к отправке: {len(rows)}")
        sent = failed = 0

        for alert_id, cluster_id, reason, payload, created_day in rows:
            text = format_alert(payload or {})
            print(f"\n--- алерт #{alert_id} ({reason}, {created_day}) ---")
            print(text)

            ok = False
            if args.dry_run or not args.token:
                # Dry-run: логика проверена, транспорт не задействован.
                # В демо показывается лента алертов на стенде.
                ok = True
            else:
                chat = args.chat_id or (payload or {}).get("owner_chat_id", "")
                if not chat:
                    print("  нет chat_id — пропускаю", file=sys.stderr)
                else:
                    ok = send_telegram(args.token, chat, text)

            cur.execute("""
                UPDATE knowledge_alerts
                SET status = %s, sent_at = now()
                WHERE id = %s
            """, ("sent" if ok else "failed", alert_id))
            conn.commit()
            if ok:
                sent += 1
            else:
                failed += 1

        cur.execute("""
            SELECT status, count(*) FROM knowledge_alerts GROUP BY status
        """)
        print("\nитого по статусам:", dict(cur.fetchall()))
        cur.close()

    print(f"отправлено: {sent}, ошибок: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())