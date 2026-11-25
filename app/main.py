"""FastAPI: стенд для демонстрации и проверки.

Один процесс, один воркер. Второй воркер удвоит ONNX-сессию (~180 МБ)
и на 2 ГБ RAM выбьет в OOM рядом с PostgreSQL.

Эндпоинты:
  GET  /            — веб-интерфейс (одна страница, без сборки)
  GET  /api/health  — статус компонентов
  POST /api/ask     — основной поиск
  POST /api/feedback — оценка ответа (полезно/бесполезно)
  GET  /api/metrics — агрегаты для дашборда
  GET  /api/stats   — состояние индекса

RAM: модель 29.5 МБ + сессия onnxruntime ~150 МБ + FastAPI ~40 МБ ≈ 230 МБ.
Латентность: p50 148 мс, p95 238 мс (замерено на этом сервере).
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from app.search.encoder import Encoder
from app.search.search import Searcher

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "refusal.json"

# Глобальные объекты: модель и соединение создаются ОДИН раз на процесс.
_state: dict = {}


def _dsn() -> str:
    dsn = os.environ.get("KB_DSN")
    if dsn:
        return dsn
    # Фолбэк для локального запуска: пароль из файла, который создал деплой.
    pass_file = Path("/root/kb_db_pass.txt")
    if pass_file.exists():
        pw = pass_file.read_text().strip()
        return f"host=127.0.0.1 port=5432 user=kb password={pw} dbname=kb"
    raise RuntimeError("не задан KB_DSN и нет /root/kb_db_pass.txt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка модели при старте: 1.5 с. Без этого первый запрос ждёт."""
    model_dir = os.environ.get("KB_MODEL_DIR", "/opt/kb/models/rubert-tiny2-int8")
    pepper = os.environ.get("KB_PEPPER", "dev-pepper-change-in-prod")

    _state["encoder"] = Encoder(model_dir, intra_threads=2)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}
    _state["searcher"] = Searcher(_dsn(), _state["encoder"], pepper, cfg)
    _state["config"] = cfg
    yield
    _state.clear()


app = FastAPI(title="KB — поиск ответов по корпоративным документам",
              version="0.3.0", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    user_id: str = Field(default="web-demo", max_length=64)
    acl: list[str] = Field(default_factory=lambda: ["all"])


class FeedbackRequest(BaseModel):
    query_hmac: str = Field(..., min_length=64, max_length=64)
    useful: bool


@app.get("/api/health")
async def health() -> dict:
    """Проверка живости: модель загружена, БД отвечает."""
    ok = {"api": True, "encoder": "encoder" in _state, "db": False, "index": {}}
    try:
        import psycopg
        with psycopg.connect(_dsn(), autocommit=True) as conn:
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM chunks WHERE is_active")
            ok["index"]["active_chunks"] = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM documents WHERE is_active")
            ok["index"]["active_docs"] = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM chunk_sentences")
            ok["index"]["sentences"] = cur.fetchone()[0]
            cur.close()
        ok["db"] = True
    except Exception as exc:  # noqa: BLE001
        ok["db_error"] = str(exc)[:200]
    return ok


@app.post("/api/ask")
async def ask(req: AskRequest) -> JSONResponse:
    """Основной поиск. Возвращает цитату или честный отказ."""
    s: Searcher | None = _state.get("searcher")
    if s is None:
        raise HTTPException(503, "модель не загружена")

    try:
        ans = s.search(req.query, user_id=req.user_id, acl=req.acl)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"ошибка поиска: {str(exc)[:200]}") from exc

    payload = ans.to_dict()
    # HMAC нужен интерфейсу, чтобы привязать оценку к запросу.
    # Сам текст запроса в журнал не пишется.
    from app.search.search import query_hmac
    payload["query_hmac"] = query_hmac(req.query, s.pepper)
    return JSONResponse(payload)


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest) -> dict:
    """Оценка ответа. Привязка по HMAC, без хранения текста."""
    try:
        import psycopg
        with psycopg.connect(_dsn(), autocommit=True) as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE query_logs
                SET user_feedback = %s, feedback_at = now()
                WHERE id = (
                    SELECT id FROM query_logs
                    WHERE query_hmac = %s
                    ORDER BY created_at DESC LIMIT 1
                )
                RETURNING id
            """, (1 if req.useful else -1, req.query_hmac))
            row = cur.fetchone()
            cur.close()
        return {"ok": row is not None, "log_id": row[0] if row else None}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)[:200]) from exc


@app.get("/api/metrics")
async def metrics(days: int = 30) -> dict:
    """Агрегаты для дашборда.

    Считаются ТОЛЬКО по is_simulated = false: смешивать измерение с
    демонстрационной историей нельзя, иначе цифра на слайде — фикция.
    """
    try:
        import psycopg
        with psycopg.connect(_dsn(), autocommit=True) as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    count(*) AS total,
                    count(*) FILTER (WHERE response_type = 'success') AS success,
                    count(*) FILTER (WHERE response_type = 'stale') AS stale,
                    count(*) FILTER (WHERE response_type = 'refusal') AS refusal,
                    count(*) FILTER (WHERE user_feedback = 1) AS useful,
                    count(*) FILTER (WHERE user_feedback = -1) AS useless,
                    count(*) FILTER (WHERE cache_hit) AS cache_hits,
                    percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) AS p50,
                    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95,
                    coalesce(sum(CASE WHEN response_type <> 'refusal'
                                      THEN 60 - latency_ms / 60000.0
                                      ELSE 0 END), 0) AS minutes_saved
                FROM query_logs
                WHERE created_at >= now() - make_interval(days => %s)
                  AND is_simulated = false
            """, (days,))
            r = cur.fetchone()
            total = r[0] or 0
            out = {
                "window_days": days,
                "total": total,
                "success": r[1], "stale": r[2], "refusal": r[3],
                "deflection_rate": round((r[1] + r[2]) / total, 4) if total else 0,
                "refusal_rate": round(r[3] / total, 4) if total else 0,
                "stale_rate": round(r[2] / total, 4) if total else 0,
                "useful": r[4], "useless": r[5],
                "useful_rate": (round(r[4] / (r[4] + r[5]), 4)
                                if (r[4] + r[5]) else 0),
                "cache_hits": r[6],
                "p50_ms": round(float(r[7]), 1) if r[7] is not None else 0,
                "p95_ms": round(float(r[8]), 1) if r[8] is not None else 0,
                "hours_saved": round(float(r[9]) / 60.0, 1),
            }
            cur.execute("""
                SELECT refusal_reason, count(*)
                FROM query_logs
                WHERE created_at >= now() - make_interval(days => %s)
                  AND response_type = 'refusal' AND is_simulated = false
                GROUP BY refusal_reason ORDER BY count(*) DESC
            """, (days,))
            out["refusal_reasons"] = {k or "unknown": v for k, v in cur.fetchall()}
            cur.close()
        return out
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)[:200]) from exc


@app.get("/api/stats")
async def stats() -> dict:
    """Состояние индекса и конфигурации — для панели и отладки."""
    cfg = _state.get("config", {})
    out = {
        "threshold": cfg.get("threshold"),
        "weights": cfg.get("weights"),
        "calibrated_on": cfg.get("calibrated_on"),
        "model": "cointegrated/rubert-tiny2 int8",
        "embedding_dim": 312,
    }
    try:
        import psycopg
        with psycopg.connect(_dsn(), autocommit=True) as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT d.doc_key, d.title, d.version, d.valid_from, d.valid_to,
                       d.is_active, count(c.id) AS chunks
                FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id ORDER BY d.is_active DESC, d.title
            """)
            out["documents"] = [{
                "doc_key": r[0], "title": r[1], "version": r[2],
                "valid_from": str(r[3]), "valid_to": str(r[4]) if r[4] else None,
                "is_active": r[5], "chunks": r[6],
            } for r in cur.fetchall()]
            cur.close()
    except Exception as exc:  # noqa: BLE001
        out["db_error"] = str(exc)[:200]
    return out


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Веб-интерфейс. Одна страница, без сборки и внешних CDN.

    Всё внутри контура: ни одного внешнего запроса. Шрифты системные,
    стили инлайн — иначе на закрытом стенде страница останется без CSS.
    """
    return HTMLResponse(INDEX_HTML)


INDEX_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Поиск по корпоративным документам</title>
<style>
  :root { --bg:#0f1115; --card:#171a21; --line:#252a34; --txt:#e6e9ef;
          --dim:#8b93a3; --ok:#2f9e6e; --warn:#c9922e; --bad:#c14b4b; --acc:#4a7fd4; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; }
  .wrap { max-width:900px; margin:0 auto; padding:28px 20px 80px; }
  h1 { font-size:21px; margin:0 0 4px; font-weight:600; }
  .sub { color:var(--dim); font-size:13px; margin-bottom:22px; }
  .row { display:flex; gap:10px; }
  input[type=text] { flex:1; padding:13px 15px; border-radius:9px; font-size:15px;
        background:var(--card); border:1px solid var(--line); color:var(--txt); }
  input[type=text]:focus { outline:none; border-color:var(--acc); }
  button { padding:13px 22px; border-radius:9px; border:0; cursor:pointer;
        background:var(--acc); color:#fff; font-size:15px; font-weight:500; }
  button:disabled { opacity:.5; cursor:default; }
  .ex { margin:14px 0 0; display:flex; flex-wrap:wrap; gap:8px; }
  .ex span { font-size:12px; color:var(--dim); padding:5px 10px; cursor:pointer;
        border:1px solid var(--line); border-radius:14px; }
  .ex span:hover { border-color:var(--acc); color:var(--txt); }
  .card { background:var(--card); border:1px solid var(--line); border-radius:11px;
        padding:18px; margin-top:18px; }
  .badge { display:inline-block; font-size:11px; padding:3px 9px; border-radius:11px;
        font-weight:600; letter-spacing:.02em; }
  .b-ok { background:rgba(47,158,110,.16); color:#5fd39b; }
  .b-warn { background:rgba(201,146,46,.16); color:#e0b055; }
  .b-bad { background:rgba(193,75,75,.16); color:#e07373; }
  .meta { color:var(--dim); font-size:12.5px; margin:9px 0 13px; }
  blockquote { margin:0; padding:13px 16px; border-left:3px solid var(--acc);
        background:rgba(74,127,212,.07); border-radius:0 8px 8px 0;
        font-size:14.5px; white-space:pre-wrap; }
  .src { font-size:12.5px; color:var(--dim); margin-top:10px; }
  .src code { color:#9db3d8; background:rgba(255,255,255,.05);
        padding:1px 6px; border-radius:4px; }
  .fb { margin-top:15px; display:flex; gap:9px; align-items:center; }
  .fb button { padding:7px 15px; font-size:13px; background:transparent;
        border:1px solid var(--line); color:var(--dim); }
  .fb button:hover { border-color:var(--acc); color:var(--txt); }
  .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
        gap:11px; margin-top:20px; }
  .stat { background:var(--card); border:1px solid var(--line);
        border-radius:10px; padding:13px; }
  .stat .v { font-size:20px; font-weight:600; }
  .stat .k { font-size:11.5px; color:var(--dim); margin-top:3px; }
  .hidden { display:none; }
  .spin { display:inline-block; width:13px; height:13px; border:2px solid var(--dim);
        border-top-color:transparent; border-radius:50%; animation:sp .7s linear infinite; }
  @keyframes sp { to { transform:rotate(360deg); } }
  .note { font-size:12px; color:var(--dim); margin-top:26px; line-height:1.6; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Поиск по корпоративным документам</h1>
  <div class="sub">Отвечает дословной цитатой со ссылкой на документ и раздел.
     Если надёжного ответа нет — отказывается.</div>

  <div class="row">
    <input type="text" id="q" placeholder="Например: кто подписывает агентский договор свыше миллиона"
           autocomplete="off">
    <button id="go">Найти</button>
  </div>
  <div class="ex" id="ex"></div>

  <div id="out"></div>

  <div class="stats" id="stats"></div>

  <div class="note" id="note"></div>
</div>

<script>
const EXAMPLES = [
  "кто подписывает агентский договор свыше миллиона",
  "какой срок согласования договора",
  "размер суточных в командировке",
  "за сколько дней писать заявление на отпуск",
  "как получить доверенность на подписание договоров",
  "как оформить командировку за границу"
];
let lastHmac = null;

const ex = document.getElementById('ex');
EXAMPLES.forEach(t => {
  const s = document.createElement('span');
  s.textContent = t;
  s.onclick = () => { document.getElementById('q').value = t; ask(); };
  ex.appendChild(s);
});

document.getElementById('go').onclick = ask;
document.getElementById('q').addEventListener('keydown', e => {
  if (e.key === 'Enter') ask();
});

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function ask() {
  const q = document.getElementById('q').value.trim();
  if (q.length < 2) return;
  const out = document.getElementById('out');
  const btn = document.getElementById('go');
  btn.disabled = true;
  out.innerHTML = '<div class="card"><span class="spin"></span> Ищу…</div>';

  try {
    const r = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: q})
    });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const d = await r.json();
    lastHmac = d.query_hmac;
    out.innerHTML = render(d);
    bindFeedback();
  } catch (e) {
    out.innerHTML = '<div class="card"><span class="badge b-bad">ОШИБКА</span> ' +
                    esc(e.message) + '</div>';
  } finally {
    btn.disabled = false;
  }
}

function render(d) {
  const isOk = d.response_type === 'success';
  const isStale = d.response_type === 'stale';
  let badge = isOk ? '<span class="badge b-ok">НАЙДЕНО В ДОКУМЕНТЕ</span>'
            : isStale ? '<span class="badge b-warn">ВЕРСИЯ УСТАРЕЛА</span>'
            : '<span class="badge b-bad">ЧЕСТНЫЙ ОТКАЗ</span>';

  let h = '<div class="card">' + badge;

  if (d.stale_warning) {
    h += '<div class="meta">' + esc(d.stale_warning) + '</div>';
  }

  if (d.sources && d.sources.length) {
    const s = d.sources[0];
    h += '<div class="meta">Документ <b>' + esc(s.doc_title) + '</b>, ' +
         'версия ' + esc(s.version) + ', раздел ' + esc(s.section_path) +
         (s.section_title ? ' «' + esc(s.section_title) + '»' : '') + '</div>';
    h += '<blockquote>' + esc(s.quote) + '</blockquote>';
    h += '<div class="src">Источник: <code>' + esc(s.source_path) + '</code></div>';
    if (d.sources.length > 1) {
      h += '<div class="src" style="margin-top:12px">Другие совпадения:</div>';
      d.sources.slice(1).forEach(o => {
        h += '<div class="src">• <b>' + esc(o.doc_title) + '</b> v' +
             esc(o.version) + ', п. ' + esc(o.section_path) + '</div>';
      });
    }
    h += '<div class="fb" data-hmac="' + esc(d.query_hmac) + '">' +
         '<span class="src" style="margin:0">Помогло?</span>' +
         '<button data-v="1">Полезно</button>' +
         '<button data-v="0">Бесполезно</button></div>';
  } else {
    h += '<div class="meta">В текущих регламентах нет прямого ответа на вопрос.</div>' +
         '<blockquote>Возможные причины: регламент устарел или случай не описан.\n' +
         'Рекомендую обратиться ' +
         (d.owner_role ? 'к специалисту: <b>' + esc(d.owner_role) + '</b>'
                       : 'к владельцу базы знаний') +
         (d.owner_contact ? ' (' + esc(d.owner_contact) + ')' : '') +
         '\nили создать заявку на обновление базы знаний.</blockquote>';
    if (d.refusal_reason) {
      h += '<div class="src">Причина: <code>' + esc(d.refusal_reason) + '</code></div>';
    }
  }

  h += '<div class="src" style="margin-top:12px">' +
       'уверенность ' + (d.confidence * 100).toFixed(0) + '% · ' +
       d.latency_ms + ' мс' + (d.cache_hit ? ' · из кэша' : '') + '</div>';
  h += '</div>';
  return h;
}

function bindFeedback() {
  document.querySelectorAll('.fb button').forEach(b => {
    b.onclick = async () => {
      const box = b.parentElement;
      const hmac = box.dataset.hmac;
      const useful = b.dataset.v === '1';
      try {
        await fetch('/api/feedback', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({query_hmac: hmac, useful})
        });
        box.innerHTML = '<span class="src" style="margin:0">Спасибо, оценка учтена.</span>';
      } catch (e) { /* молча: оценка не должна ломать интерфейс */ }
    };
  });
}

async function loadStats() {
  try {
    const [m, s] = await Promise.all([
      fetch('/api/metrics?days=30').then(r => r.json()),
      fetch('/api/stats').then(r => r.json())
    ]);
    const items = [
      [m.total, 'запросов'],
      [(m.deflection_rate * 100).toFixed(0) + '%', 'отвечено'],
      [(m.refusal_rate * 100).toFixed(0) + '%', 'отказов'],
      [m.p50_ms + ' мс', 'медиана'],
      [s.documents ? s.documents.filter(d => d.is_active).length : 0, 'документов']
    ];
    document.getElementById('stats').innerHTML = items.map(
      ([v, k]) => '<div class="stat"><div class="v">' + esc(v) +
                  '</div><div class="k">' + esc(k) + '</div></div>'
    ).join('');
    const cfgNote = s.threshold
      ? 'Порог отказа ' + Number(s.threshold).toFixed(2) +
        ', откалиброван на ' + esc(s.calibrated_on || '—') +
        '. Модель: ' + esc(s.model) + ', ' + s.embedding_dim + ' измерений.'
      : '';
    document.getElementById('note').textContent = cfgNote;
  } catch (e) { /* статистика не критична */ }
}

loadStats();
setInterval(loadStats, 30000);
</script>
</body>
</html>"""