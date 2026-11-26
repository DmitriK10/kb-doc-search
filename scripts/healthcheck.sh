#!/bin/bash
# ============================================================================
# healthcheck.sh — проверка доступности стенда с уведомлением на почту
#
# ЗАЧЕМ. Бриф (ответы 8-12): стенд должен быть в изолированном контуре и
# не делать внешних вызовов. Это относится и к мониторингу: проверка идёт
# локально (curl на 127.0.0.1), письмо уходит в локальный SMTP (mailpit).
# Никаких внешних сервисов мониторинга.
#
# ПОЧЕМУ СВОЙ СКРИПТ, А НЕ uptime-сервис. Внешний сервис потребовал бы
# исходящих вызовов и учётной записи — прямое нарушение требований контура.
# Здесь всё локально и проверяемо.
#
# ЧТО ПРОВЕРЯЕТ:
#   1. kb-api отвечает на /api/health (не просто процесс жив, а БД и индекс)
#   2. nginx отдаёт /kb/ по HTTPS
#   3. свободная память не ниже порога (на 2 ГБ RSS утекает незаметно)
#
# КАК ЧАСТО: cron каждые 5 минут. При падении — письмо, но не чаще раза
# в 30 минут (чтобы не забить ящик при долгом сбое).
# ============================================================================

set -uo pipefail

PASS_FILE=/root/kb_db_pass.txt
STATE_DIR=/var/lib/kb-health
STATE_FILE="$STATE_DIR/last_alert"
LOCK_FILE="$STATE_DIR/lock"
mkdir -p "$STATE_DIR"

API_URL="http://127.0.0.1:8090/api/health"
SITE_URL="https://vds3095790.my-ihor.ru/kb/"
SMTP_HOST=127.0.0.1
SMTP_PORT=1025
MAIL_TO="quality@company.local"
# Порог свободной памяти: 200 МБ. Ниже — риск OOM для kb-api (RSS 160 МБ).
MIN_FREE_MB=200
# Не чаще одного письма в 30 минут.
ALERT_INTERVAL=1800

problems=()

# --- 1. API и его зависимости ---
api_json=$(curl -s --max-time 5 "$API_URL" 2>/dev/null)
if [ -z "$api_json" ]; then
    problems+=("kb-api не отвечает на /api/health")
elif ! echo "$api_json" | grep -q '"db":true'; then
    problems+=("kb-api: база данных недоступна")
elif ! echo "$api_json" | grep -q '"encoder":true'; then
    problems+=("kb-api: энкодер не загружен")
fi

# --- 2. nginx и HTTPS ---
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "$SITE_URL" 2>/dev/null)
if [ "$code" != "200" ]; then
    problems+=("nginx: /kb/ вернул код $code вместо 200")
fi

# --- 3. Память ---
free_mb=$(free -m | awk 'NR==2{print $7}')
if [ -n "$free_mb" ] && [ "$free_mb" -lt "$MIN_FREE_MB" ]; then
    problems+=("мало памяти: свободно ${free_mb} МБ (порог ${MIN_FREE_MB})")
fi

# --- Решение: молчать или писать ---
if [ ${#problems[@]} -eq 0 ]; then
    # Всё в порядке: если раньше было письмо о сбое, сообщаем о восстановлении.
    if [ -f "$STATE_FILE" ]; then
        printf 'Subject: Стенд снова доступен\n\nВсе проверки прошли:\n  API: ок\n  HTTPS: 200\n  Память: %s МБ свободно\n' \
            "$free_mb" | /opt/kb/.venv/bin/python3 - "$MAIL_TO" "$SMTP_HOST" "$SMTP_PORT" 2>/dev/null \
            || echo "восстановление: не удалось отправить"
        rm -f "$STATE_FILE"
    fi
    exit 0
fi

# Есть проблема. Проверяем, не писали ли недавно.
now=$(date +%s)
if [ -f "$STATE_FILE" ]; then
    last=$(cat "$STATE_FILE" 2>/dev/null || echo 0)
    if [ $((now - last)) -lt "$ALERT_INTERVAL" ]; then
        exit 1
    fi
fi
echo "$now" > "$STATE_FILE"

body=$(printf '%s\n' "${problems[@]}")
printf 'Subject: Стенд kb: проблема\n\nПроверка в %s MSK.\n\nПроблемы:\n%s\n\nПроверить:\n  systemctl status kb-api\n  systemctl status nginx\n  journalctl -u kb-api -n 50\n' \
    "$(date '+%Y-%m-%d %H:%M')" "$body" > /tmp/kb_alert.txt

# Отправка через локальный SMTP средствами Python (smtplib есть в venv).
/opt/kb/.venv/bin/python - <<'PY' 2>/dev/null
import smtplib
from email.message import EmailMessage
from email.utils import formatdate
from pathlib import Path

raw = Path("/tmp/kb_alert.txt").read_text(encoding="utf-8")
subject, _, body = raw.partition("\n\n")
subject = subject.removeprefix("Subject: ").strip()

msg = EmailMessage()
msg["From"] = "kb-health@company.local"
msg["To"] = "quality@company.local"
msg["Subject"] = subject
msg["Date"] = formatdate(localtime=True)
msg.set_content(body)

with smtplib.SMTP("127.0.0.1", 1025, timeout=10) as s:
    s.send_message(msg)
PY

exit 1