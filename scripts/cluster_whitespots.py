"""Кластеризация отказов: поиск «белых пятен» в базе знаний.

Запускается по таймеру раз в час, живёт 2-5 секунд, потребляет ~30 МБ.
НЕ на лету: кластеризация в обработчике запроса добавила бы 200+ мс и
проблемы с конкурентностью.

Что делает:
  1. Берёт эмбеддинги отказов за окно (7 дней) + «бесполезные» ответы.
  2. Кластеризует KMeans на numpy (без sklearn — 80 МБ ради одного вызова).
  3. Сопоставляет центроиды с прошлым прогоном по косинусу: stable_key
     наследуется. Без этого Grafana каждый час показывает «новые» красные
     зоны, и владельцы перестают реагировать на алерты.
  4. Метку кластера берёт из ПРОСТРАНСТВА ДОКУМЕНТОВ (ближайшие чанки к
     центроиду), а не из текста запроса: сырой текст не хранится (152-ФЗ).
     Это не компромисс, а рабочее решение — ближайший чанк даёт и тему,
     и владельца.
  5. Считает зоны: red при доле отказов >= 0.6, yellow при доле
     «бесполезно» >= 0.3, иначе green.
  6. Пишет алерты в knowledge_alerts с дедупликацией за день.

RAM: центроиды 312 × k float32, при k=12 это 15 КБ. Матрица эмбеддингов
при 300 отказах — 375 КБ. Ничтожно.
Скорость: KMeans на 300×312, k=12, 40 итераций — ~120 мс на 2 vCPU.

Запуск:
    python scripts/cluster_whitespots.py --dsn "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.search.encoder import to_pgvector  # noqa: E402
from app.utils.vec import normalize_rows, to_f32  # noqa: E402

# Пороги зон. Зафиксированы здесь, а не в SQL, потому что это бизнес-правило,
# которое обсуждается с владельцем базы знаний.
RED_REFUSAL_SHARE = 0.60
YELLOW_USELESS_SHARE = 0.30

# Минимальный размер кластера: кластер из 1-2 запросов — это шум, а не
# «белое пятно». Без порога дашборд заполняется одноразовыми темами.
MIN_CLUSTER_SIZE = 3

# Окно анализа: 7 дней. Меньше — мало данных, больше — зоны не отражают
# текущее состояние базы.
WINDOW_DAYS = 7

# Порог сопоставления центроидов между прогонами.
STABLE_SIM = 0.90


def sq_dists(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Квадраты евклидовых расстояний (n, k).

    Обрезка по нулю обязательна. Формула |x|^2 - 2xc + |c|^2 на float32
    даёт небольшие ОТРИЦАТЕЛЬНЫЕ значения из-за потери точности при
    вычитании близких больших чисел. Последствия были реальными:
      - k-means++ падал с «Probabilities are not non-negative»;
      - argmin выбирал «самую отрицательную» точку как самую далёкую,
        то есть ровно наоборот.
    """
    d = (x * x).sum(1)[:, None] - 2.0 * x @ c.T + (c * c).sum(1)[None, :]
    return np.maximum(d, 0.0)


def kmeans(
    x: np.ndarray, k: int, iters: int = 40, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """KMeans с инициализацией k-means++.

    Своя реализация вместо sklearn: sklearn тянет scipy (~80 МБ RSS),
    а нужен один вызов раз в час. Здесь 40 строк и никаких зависимостей.

    Пустые кластеры пересеваем в самую дальнюю точку: иначе центроид
    остаётся нулевым и следующий прогон даёт другие зоны.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return np.zeros((0, x.shape[1]), dtype=np.float32), np.zeros(0, dtype=int)

    k = max(1, min(k, n))

    # k-means++: первый центр случайный, каждый следующий — с вероятностью,
    # пропорциональной квадрату расстояния до ближайшего центра.
    c = x[rng.integers(n)][None, :].copy()
    for _ in range(1, k):
        d = sq_dists(x, c).min(axis=1)
        total = d.sum()
        if total <= 0:
            c = np.vstack([c, x[rng.integers(n)]])
            continue
        c = np.vstack([c, x[rng.choice(n, p=d / total)]])

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        labels = sq_dists(x, c).argmin(axis=1)
        moved = False
        for j in range(k):
            mask = labels == j
            if mask.any():
                new = x[mask].mean(axis=0)
                if not np.allclose(new, c[j]):
                    c[j] = new
                    moved = True
            else:
                far = sq_dists(x, c).min(axis=1).argmax()
                c[j] = x[far]
                moved = True
        if not moved:
            break

    labels = sq_dists(x, c).argmin(axis=1)
    return c.astype(np.float32), labels


def choose_k(n: int) -> int:
    """Правило выбора k: стабильность важнее оптимальности.

    k = clip(round(sqrt(n/2)), 3, 12). При 300 точках это 12, при 20 — 3.
    Слишком большое k дробит зоны на одноразовые темы, слишком малое
    смешивает разные темы в одну зону — обе крайности делают дашборд
    бесполезным.
    """
    return int(max(3, min(12, round((n / 2) ** 0.5))))


def zone_for(refusal_share: float, useless_share: float) -> str:
    """Зона кластера.

    red    — много отказов: нет актуального регламента.
    yellow — отвечаем, но помечают «бесполезно»: регламент есть, но не
             отвечает на вопрос или устарел по содержанию.
    green  — отвечаем уверенно и полезно.
    """
    if refusal_share >= RED_REFUSAL_SHARE:
        return "red"
    if useless_share >= YELLOW_USELESS_SHARE:
        return "yellow"
    return "green"


def main() -> int:
    ap = argparse.ArgumentParser(description="Кластеризация белых пятен")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    ap.add_argument("--min-size", type=int, default=MIN_CLUSTER_SIZE)
    ap.add_argument("--notify", action="store_true",
                    help="создать алерты владельцам (иначе только запись)")
    args = ap.parse_args()

    import psycopg

    conn = psycopg.connect(args.dsn, autocommit=False)
    cur = conn.cursor()

    # --- 1. Собираем векторы проблемных запросов ---
    #
    # Берём и отказы, и «бесполезно»: и то, и другое — сигнал, что база
    # не отвечает на реальный вопрос. Отказы дают красные зоны,
    # «бесполезно» — жёлтые.
    cur.execute("""
        SELECT id, embedding, response_type,
               coalesce(user_feedback, 0) AS fb
        FROM query_logs
        WHERE created_at >= now() - make_interval(days => %s)
          AND (response_type = 'refusal' OR user_feedback = -1)
    """, (args.window_days,))
    rows = cur.fetchall()

    if len(rows) < args.min_size:
        print(f"недостаточно данных: {len(rows)} < {args.min_size}")
        cur.close()
        conn.close()
        return 0

    ids = [r[0] for r in rows]
    mat = np.vstack([to_f32(r[1]) for r in rows])
    mat = normalize_rows(mat)
    is_refusal = np.array([1 if r[2] == "refusal" else 0 for r in rows])
    is_useless = np.array([1 if r[3] == -1 else 0 for r in rows])

    print(f"проблемных запросов за {args.window_days} дн.: {len(rows)} "
          f"(отказов {int(is_refusal.sum())}, бесполезно {int(is_useless.sum())})")

    # --- 2. Кластеризация ---
    k = choose_k(len(rows))
    centroids, labels = kmeans(mat, k)
    print(f"k = {k}")

    # --- 3. Сопоставление с прошлым прогоном (стабильные ключи) ---
    cur.execute("""
        SELECT stable_key, centroid, status FROM query_clusters
        WHERE run_id = (SELECT max(run_id) FROM cluster_runs)
    """)
    prev = cur.fetchall()
    prev_keys = [r[0] for r in prev]
    prev_vecs = np.vstack([to_f32(r[1]) for r in prev]) if prev else None
    prev_status = {r[0]: r[2] for r in prev}

    # --- 4. Новый прогон ---
    cur.execute("""
        INSERT INTO cluster_runs (window_from, k, n_vectors)
        VALUES (now() - make_interval(days => %s), %s, %s)
        RETURNING run_id
    """, (args.window_days, k, len(rows)))
    run_id = cur.fetchone()[0]

    created = 0
    for j in range(len(centroids)):
        mask = labels == j
        size = int(mask.sum())
        if size < args.min_size:
            continue

        centroid = centroids[j]
        refusal_share = float(is_refusal[mask].mean())
        useless_share = float(is_useless[mask].mean())
        zone = zone_for(refusal_share, useless_share)

        # stable_key: ищем прошлый центроид с косинусом >= 0.90
        stable_key = None
        if prev_vecs is not None and len(prev_vecs):
            sims = normalize_rows(prev_vecs) @ (
                centroid / max(float(np.linalg.norm(centroid)), 1e-9)
            )
            best = int(np.argmax(sims))
            if float(sims[best]) >= STABLE_SIM:
                stable_key = prev_keys[best]
        if stable_key is None:
            # Новый кластер: ключ из ближайшего документа + номер.
            stable_key = f"cluster_{run_id}_{j}"

        # Метка из пространства документов: ближайшие чанки к центроиду.
        # Сырой текст запроса не хранится, поэтому имя темы берём из
        # документов — это и есть «о чём люди спрашивают и не находят».
        cur.execute("""
            SELECT c.section_title, c.h1_path, c.section_path, c.body,
                   d.title, d.owner_role, d.owner_contact,
                   1 - (c.embedding <=> %s::halfvec(312)) AS sim
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.is_active
            ORDER BY c.embedding <=> %s::halfvec(312)
            LIMIT 3
        """, (to_pgvector(centroid), to_pgvector(centroid)))
        near = cur.fetchall()

        # Метка и владелец — из ОНТОЛОГИИ, а не из ближайшего документа.
        #
        # Почему не документ. Центроид ОТКАЗОВ лежит вне пространства
        # документов: отказ означает «в базе этого нет». Поэтому ближайший
        # чанк — случайный сосед, и кластер отказов про доверенности
        # подписывался «Положение о командировках: Суточные» (проверено).
        # Владелец при этом тоже оказывался не тот, и алерт уходил не туда.
        #
        # Онтология (knowledge_owners) — это заранее описанные темы с
        # владельцами, включая темы, которых в базе НЕТ. Именно они и нужны:
        # «Оформление доверенностей → Корпоративный юрист». Это ровно то,
        # что требуется в шаблоне отказа («обратитесь к [должность]»).
        cur.execute("""
            SELECT topic_key, title, role, contact,
                   1 - (embedding <=> %s::halfvec(312)) AS sim
            FROM knowledge_owners
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::halfvec(312)
            LIMIT 1
        """, (to_pgvector(centroid), to_pgvector(centroid)))
        owner_hit = cur.fetchone()

        if owner_hit and float(owner_hit[4]) >= 0.30:
            label = owner_hit[1]
            owner_role = owner_hit[2]
            owner_contact = owner_hit[3]
        elif near:
            # Онтология не покрывает тему — показываем ближайший документ
            # и честно помечаем, что это НЕ тема запроса, а соседство.
            label = f"Тема вне онтологии (ближайшее: {near[0][4]})"
            owner_role = near[0][5]
            owner_contact = near[0][6]
        else:
            label, owner_role, owner_contact = "Без темы", None, None

        if near:
            top = near[0]
            nearest = [{
                "doc_title": r[4], "section_path": r[2],
                "section_title": r[0], "sim": round(float(r[7]), 4),
            } for r in near]
        else:
            label, owner_role, owner_contact, nearest = "Без темы", None, None, []

        cur.execute("""
            INSERT INTO query_clusters
                (run_id, stable_key, centroid, size, refusal_share,
                 useless_share, zone, label, nearest_chunks, owner_role, status)
            VALUES (%s, %s, %s::halfvec, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
            RETURNING id
        """, (
            run_id, stable_key, to_pgvector(centroid), size, refusal_share,
            useless_share, zone, label[:200], json.dumps(nearest, ensure_ascii=False),
            owner_role, prev_status.get(stable_key, "new"),
        ))
        cluster_id = cur.fetchone()[0]
        created += 1

        print(f"  [{zone:6s}] size={size:3d} refusal={refusal_share:.2f} "
              f"useless={useless_share:.2f} key={stable_key[:24]:24s} {label[:42]}")

        # --- 5. Алерты владельцам ---
        # Правило из ТЗ: >=3 отказов ИЛИ >=3 «бесполезно» по одной теме.
        # Дедупликация через UNIQUE (cluster_id, reason, created_day).
        if args.notify and zone in ("red", "yellow"):
            reason = "refusals_3" if zone == "red" else "useless_3"
            payload = {
                "cluster": stable_key,
                "topic": label,
                "zone": zone,
                "size": size,
                "refusal_share": round(refusal_share, 3),
                "useless_share": round(useless_share, 3),
                "nearest_doc": nearest[0]["doc_title"] if nearest else None,
                "owner_role": owner_role,
                "owner_contact": owner_contact,
            }
            cur.execute("""
                INSERT INTO knowledge_alerts (cluster_id, reason, payload)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT DO NOTHING
            """, (cluster_id, reason, json.dumps(payload, ensure_ascii=False)))

    conn.commit()

    # Итог
    cur.execute("""
        SELECT zone, count(*), sum(size)
        FROM query_clusters WHERE run_id = %s GROUP BY zone ORDER BY zone
    """, (run_id,))
    summary = cur.fetchall()
    print(f"\nпрогон {run_id}: создано кластеров {created}")
    for zone, n, total in summary:
        print(f"  {zone:6s}: {n} кластеров, {total} запросов")

    cur.execute("""
        SELECT count(*) FROM knowledge_alerts WHERE status = 'pending'
    """)
    print(f"алертов ожидает отправки: {cur.fetchone()[0]}")

    cur.close()
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())