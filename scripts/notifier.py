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


def format_email(request: dict, payload: dict) -> tuple[str, str]:
    """Тема и тело письма владельцу.

    Требование п. 3: номер запроса, текст вопроса, рекомендация.
    Формат — рабочий документ, а не «уведомление системы»: владелец
    должен понять за 10 секунд, что спрашивают и что делать.
    """
    qid = request.get("public_id") or "—"
    question = request.get("question") or "—"
    # Тема: из payload (алерты кластеров) или из ближайшего найденного
    # документа. Без этого тема была «тема без названия» — владелец
    # не понимал, о чём письмо, не открыв его.
    topic = payload.get("topic") or ""
    if not topic:
        closest = request.get("closest_chunks") or []
        if closest:
            topic = closest[0].get("doc_title") or ""
    if not topic:
        # Последний резерв — сам вопрос, обрезанный по слову.
        topic = (question[:60].rsplit(" ", 1)[0] + "…"
                 if len(question) > 60 else question)
    size = payload.get("size", 0)
    owner = payload.get("owner_role") or "владелец не задан"

    subject = f"[{qid}] Нет ответа в базе знаний: {topic[:60]}"

    body_parts = [
        f"Запрос № {qid}",
        f"Владелец темы: {owner}",
        "",
        "ВОПРОС СОТРУДНИКА",
        question,
        "",
        "ЧТО ПРОВЕРЕНО АГЕНТОМ",
    ]
    closest = request.get("closest_chunks") or []
    if closest:
        for c in closest[:3]:
            body_parts.append(
                f"  - {c.get('doc_title', '')}, п. {c.get('section_path', '')}: "
                f"{(c.get('quote') or '')[:110]}"
            )
        body_parts.append("")
        body_parts.append(
            "Ни один из найденных фрагментов не отвечает на вопрос "
            "с достаточной уверенностью."
        )
    else:
        body_parts.append("  релевантных фрагментов не найдено")

    body_parts += [
        "",
        "ЧТО НУЖНО СДЕЛАТЬ",
        "1. Проверить, есть ли действующий регламент по теме.",
        "2. Если есть — загрузить в базу знаний или дать ответ на этот запрос.",
        "3. Если нет — создать документ.",
        "",
        f"За период по теме «{topic}» было {size} запросов без ответа.",
        "",
        "-- ",
        "Агент поиска по корпоративным документам",
        "(письмо сформировано автоматически, отвечать на него не нужно)",
    ]
    return subject, "\n".join(body_parts)


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


def send_smtp(
    host: str, port: int, to_addr: str, subject: str, text: str,
    from_addr: str = "kb-agent@company.local",
) -> bool:
    """Отправка уведомления по электронной почте.

    Почта — основной канал заказчика: их же регламенты предписывают
    уведомлять по email (Проектная ДИБ п. 1.3, п. 2.1, п. 3.1, п. 7.3.1;
    ДБРЦТ п. 7.1). Поэтому SMTP — дефолт, Telegram остаётся fallback.

    В контуре: локальный SMTP, внешних вызовов нет.
    """
    import smtplib
    from email.message import EmailMessage
    from email.utils import formatdate

    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=True)
    msg.set_content(text)

    try:
        with smtplib.SMTP(host, port, timeout=10) as smtp:
            smtp.send_message(msg)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  SMTP ошибка: {exc}", file=sys.stderr)
        return False


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
    # SMTP — ДЕФОЛТ. Опора: регламенты заказчика предписывают почту
    # (Проектная ДИБ п. 1.3, 2.1, 3.1, 7.3.1; ДБРЦТ п. 7.1).
    ap.add_argument("--smtp-host", default="127.0.0.1")
    ap.add_argument("--smtp-port", type=int, default=1025)
    # Получателей может быть несколько: владелец темы, служба качества.
    # ВАЖНО: адрес организаторов хакатона в получатели не ставим — это
    # демонстрационный контур, а не их почта.
    ap.add_argument("--email-to", default="", help="адреса через запятую")
    ap.add_argument("--dry-run", action="store_true",
                    help="не отправлять, только пометить sent и напечатать")
    # Telegram — fallback, включается только явно.
    ap.add_argument("--token", default="", help="Telegram bot token (fallback)")
    ap.add_argument("--chat-id", default="", help="chat_id получателя")
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

        print(f"алертов к отправке: {len(rows)}")

        # Запросы в службу качества отправляются почтой: это прямой
        # канал «вопрос -> владелец» (п. 3 брифинга).
        cur.execute("""
            SELECT public_id, question, closest_chunks, confidence, coverage
            FROM quality_requests
            WHERE status = 'open'
            ORDER BY created_at DESC
            LIMIT %s
        """, (args.limit,))
        q_rows = cur.fetchall()

        sent = failed = 0
        for q in q_rows:
            public_id, question, closest, conf, cov = q
            request = {
                "public_id": public_id, "question": question,
                "closest_chunks": closest or [],
            }
            subject, body = format_email(request, {})
            print(f"\n--- запрос {public_id} ---")
            print(subject)
            print(body[:400])

            ok = False
            if args.dry_run:
                ok = True
            elif args.email_to:
                ok = send_smtp(args.smtp_host, args.smtp_port,
                               ", ".join(to_list), subject, body)
            elif args.token:
                ok = send_telegram(args.token, args.chat_id, body)
            else:
                print("  нет ни --email-to, ни --token — пропускаю",
                      file=sys.stderr)

            if ok and not args.dry_run:
                cur.execute("""
                    UPDATE quality_requests SET status = 'answered'
                    WHERE public_id = %s AND status = 'open'
                """, (public_id,))
                # ВНИМАНИЕ: статус 'answered' здесь означает «отправлено
                # владельцу», а не «получен ответ». Ответ приходит через
                # /api/quality/answer и переводит в 'answered' повторно.
                cur.execute("""
                    UPDATE quality_requests SET status = 'open'
                    WHERE public_id = %s AND answer IS NULL
                """, (public_id,))
            if ok:
                sent += 1
            else:
                failed += 1

        # Сигналы о белых пятнах — тем же путём.
        for alert_id, cluster_id, reason, payload, created_day in rows:
            text = format_alert(payload or {})
            print(f"\n--- алерт #{alert_id} ({reason}, {created_day}) ---")
            print(text[:300])

            ok = False
            if args.dry_run:
                ok = True
            elif args.email_to:
                topic = (payload or {}).get("topic", "белое пятно")
                ok = send_smtp(
                    args.smtp_host, args.smtp_port, ", ".join(to_list),
                    f"Белое пятно в базе знаний: {topic[:60]}", text,
                )
            elif args.token:
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