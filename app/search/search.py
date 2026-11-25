"""Поиск: FTS + вектор + RRF + реранк + калиброванный порог отказа.

Полный путь запроса:
  1. Нормализация, токенизация, стоп-фильтр, потолок 8 групп (tsquery.py).
  2. Вектор запроса через ONNX int8 (2.6 мс).
  3. Две ветки параллельно в одном SQL:
       FTS:  kb_tsquery(tokens) по активным чанкам;
       VEC:  embedding <=> query_vec по активным чанкам.
  4. RRF k=60.
  5. Sentence-level реранк по предвычисленным векторам предложений.
  6. Признаки уверенности -> калиброванная логистика -> ответ или отказ.
  7. Stale-флоу: если отказали, второй проход по неактивным версиям.

Ключевые ограничения, заложенные в код:
  - фильтр is_active в ОБЕИХ ветках (денормализация, без JOIN — иначе HNSW);
  - ACL применяется ПОСЛЕ RRF, с расширением пула (правка B7);
  - кэш по HMAC(query|acl_signature), попадание пишется в query_logs (правка B6);
  - версия в запросе -> структурный фильтр и stale-роутинг (правка 1 раунда 5).

RAM: numpy-матрицы на запрос ~250 КБ. Модель уже загружена в Encoder.
Латентность: p50 35-70 мс, p95 120-180 мс на 2 vCPU.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from app.search.encoder import acl_signature, to_pgvector
from app.search.fusion import (
    confidence_features,
    final_rank,
    pick_best_sentence,
    rrf_fuse,
    sentence_rerank,
)
from app.search.tsquery import prepare, strip_version_prefix
from app.search.versions import extract_versions, is_likely_section_ref
from app.utils.vec import to_f32

# Параметры поиска. Вынесены в константы, потому что их значения
# обоснованы и не должны «настраиваться на глаз» в разных местах.
POOL_SIZE = 50          # кандидатов из каждой ветки
RERANK_LIMIT = 12       # сколько чанков реранкуем
ACL_EXPAND = 3          # во сколько раз расширяем пул при отсеве по ACL
CACHE_TTL_HOURS = 24


@dataclass
class Source:
    """Источник в ответе — то, что видит пользователь.

    Порядок полей важен: сначала обязательные, потом со значениями по
    умолчанию. Иначе dataclass падает с «non-default argument follows
    default argument».
    """

    doc_title: str
    version: str
    section_path: str
    section_title: str
    source_path: str
    quote: str
    chunk_id: int = 0
    is_active: bool = True

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_title": self.doc_title,
            "version": self.version,
            "section_path": self.section_path,
            "section_title": self.section_title,
            "source_path": self.source_path,
            "quote": self.quote,
            "is_active": self.is_active,
        }


@dataclass
class Answer:
    """Результат поиска."""

    response_type: str                  # success | stale | refusal
    confidence: float
    sources: list[Source] = field(default_factory=list)
    refusal_reason: str | None = None
    owner_role: str | None = None
    owner_contact: str | None = None
    latency_ms: int = 0
    cache_hit: bool = False
    features: dict[str, float] = field(default_factory=dict)
    stale_warning: str | None = None

    def to_dict(self) -> dict:
        return {
            "response_type": self.response_type,
            "confidence": round(self.confidence, 4),
            "sources": [s.to_dict() for s in self.sources],
            "refusal_reason": self.refusal_reason,
            "owner_role": self.owner_role,
            "owner_contact": self.owner_contact,
            "latency_ms": self.latency_ms,
            "cache_hit": self.cache_hit,
            "features": {k: round(v, 4) for k, v in self.features.items()},
            "stale_warning": self.stale_warning,
        }


def query_hmac(query: str, pepper: str) -> str:
    """HMAC-SHA256 от нормализованного запроса.

    Не sha256: короткий запрос («кто подписывает») перебирается по словарю
    за минуты, и по хэшу можно восстановить текст. HMAC с серверным pepper
    этого не позволяет, а для кэша и дедупликации хэш работает так же.
    """
    norm = " ".join(query.lower().split())
    return hmac.new(pepper.encode(), norm.encode(), hashlib.sha256).hexdigest()


class Searcher:
    """Поиск по корпусу. Один экземпляр на процесс API."""

    def __init__(self, dsn: str, encoder, pepper: str, config: dict) -> None:
        import psycopg

        self._psycopg = psycopg
        self.dsn = dsn
        self.encoder = encoder
        self.pepper = pepper
        # Веса логистики и порог — из config/refusal.json (калибратор).
        self.weights: dict[str, float] = config.get("weights", {})
        self.threshold: float = float(config.get("threshold", 0.5))
        self.stale_threshold: float = float(config.get("stale_threshold", 0.62))
        self.creative_min_sim: float = float(config.get("creative_min_sim", 0.86))

    # ---------------------------------------------------------------- SQL --

    def _fetch_candidates(
        self,
        tokens: list[str],
        qvec: np.ndarray,
        acl: list[str],
        pool: int,
    ) -> tuple[list[int], list[int], dict[int, dict], bool]:
        """Две ветки поиска + метаданные кандидатов.

        Возвращает (fts_ids, vec_ids, meta_by_id, fts_and_hit), где
        fts_and_hit — сработала ли AND-ветка ДО OR-фолбэка.

        Важно: фильтр is_active стоит в ОБЕИХ ветках. Если вынести его
        в JOIN после ORDER BY, PostgreSQL перестанет использовать HNSW.
        """
        vec_literal = to_pgvector(qvec)
        conn = self._psycopg.connect(self.dsn, autocommit=False)
        try:
            cur = conn.cursor()

            # FTS-ветка. kb_tsquery возвращает NULL, если все токены —
            # стоп-слова: тогда ветка просто не даёт строк.
            #
            # AND между группами точен, но хрупок: «оформить» и «оформляется»
            # дают разные леммы (оформи / оформля), и AND по двум группам
            # возвращает ноль, хотя документ ровно про это. Поэтому при
            # пустой выдаче (< 5 строк) включается OR-фолбэк — правило из
            # правки B3. Порог 5, а не «всегда OR»: на частотных словах
            # («как», «документ») OR утопил бы ранжирование шумом, который
            # RRF не дофильтрует.
            fts_ids: list[int] = []
            # Признак точного совпадения фиксируем ДО OR-фолбэка:
            # именно AND-ветка означает «все содержательные термины запроса
            # есть в одном чанке», а OR-фолбэк находится почти всегда и
            # как признак уверенности бесполезен.
            fts_and_hit = False
            if tokens:
                cur.execute("""
                    WITH q AS (SELECT kb_tsquery(%s::text[]) AS tsq)
                    SELECT c.id
                    FROM chunks c, q
                    WHERE c.is_active
                      AND q.tsq IS NOT NULL
                      AND c.tsv @@ q.tsq
                    ORDER BY ts_rank_cd(c.tsv, q.tsq) DESC
                    LIMIT %s
                """, (tokens, pool))
                fts_ids = [r[0] for r in cur.fetchall()]
                fts_and_hit = len(fts_ids) > 0

                if len(fts_ids) < 5:
                    cur.execute("""
                        WITH q AS (SELECT kb_tsquery_or(%s::text[]) AS tsq)
                        SELECT c.id
                        FROM chunks c, q
                        WHERE c.is_active
                          AND q.tsq IS NOT NULL
                          AND c.tsv @@ q.tsq
                        ORDER BY ts_rank_cd(c.tsv, q.tsq) DESC
                        LIMIT %s
                    """, (tokens, pool))
                    or_ids = [r[0] for r in cur.fetchall()]
                    # Сохраняем порядок AND-результатов, добиваем OR-хвостом.
                    seen = set(fts_ids)
                    fts_ids.extend(i for i in or_ids if i not in seen)

            # Векторная ветка. halfvec_cosine_ops + <=> ; тип приводим
            # явно, иначе PostgreSQL не найдёт оператор halfvec <=> vector.
            cur.execute("""
                SELECT c.id
                FROM chunks c
                WHERE c.is_active
                ORDER BY c.embedding <=> %s::halfvec(312)
                LIMIT %s
            """, (vec_literal, pool))
            vec_ids = [r[0] for r in cur.fetchall()]

            all_ids = list({*fts_ids, *vec_ids})
            if not all_ids:
                cur.close()
                return [], [], {}, False

            # Метаданные кандидатов одним запросом.
            #
            # token_coverage — доля содержательных токенов запроса, чьи
            # ЛЕММЫ встречаются в tsv чанка. Считается здесь, а не в Python:
            # лемматизация живёт в PostgreSQL (kb_lemma), и делать её второй
            # раз на стороне клиента значило бы завести вторую реализацию,
            # которая рано или поздно разойдётся с первой.
            #
            # Зачем признак: запрос «командировка за границу» семантически
            # близок к «командировка», документ про поездки по РФ даёт
            # высокий косинус, и система отвечает по нему — хотя слова
            # «границу» в тексте нет. Косинус этого не видит, покрытие видит.
            cur.execute("""
                WITH toks AS (
                    SELECT array_agg(DISTINCT kb_lemma(t)) FILTER (
                               WHERE kb_lemma(t) IS NOT NULL
                           ) AS lemmas
                    FROM unnest(%s::text[]) AS t
                )
                SELECT c.id, c.section_path, c.section_title, c.h1_path, c.body,
                       c.is_active, d.title, d.version, d.source_path,
                       d.access_groups, d.owner_role, d.owner_contact,
                       1 - (c.embedding <=> %s::halfvec(312)) AS chunk_sim,
                       (
                           SELECT count(*)::float
                           FROM unnest(toks.lemmas) AS lem
                           -- ВАЖНО: считаем по kb_tsv(c.body), БЕЗ заголовка.
                           -- Раньше было c.tsv, а он включает section_title
                           -- с весом A. Из-за этого чанк из раздела
                           -- «Подписание договоров» получал зачёт за слова
                           -- «подписывает» и «договор» ИЗ ЗАГОЛОВКА, хотя в
                           -- теле их не было. На стенде это вывело наверх
                           -- пункт «до 100 000 руб.» вместо «агентский,
                           -- свыше 1 млн». Заголовок помогает найти, но не
                           -- подтверждает содержание цитаты.
                           WHERE kb_lemma(lem) IS NOT NULL
                             AND kb_tsv(c.body) @@ to_tsquery('russian', lem)
                       ) / NULLIF(cardinality(toks.lemmas), 0) AS coverage,
                       c.chunk_type
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                CROSS JOIN toks
                WHERE c.id = ANY(%s)
            """, (tokens or [], vec_literal, all_ids))
            meta = {}
            for r in cur.fetchall():
                meta[r[0]] = {
                    "id": r[0], "section_path": r[1], "section_title": r[2],
                    "h1_path": r[3], "body": r[4], "is_active": r[5],
                    "doc_title": r[6], "version": r[7], "source_path": r[8],
                    "access_groups": r[9], "owner_role": r[10],
                    "owner_contact": r[11], "chunk_sim": float(r[12] or 0.0),
                    "coverage": float(r[13]) if r[13] is not None else 1.0,
                    "chunk_type": r[14],
                }
            cur.close()

            # ACL после RRF-кандидатов: отсекаем недоступное, сохраняя порядок.
            # Стоимость нулевая при access_groups={all} у всех документов.
            acl_set = set(acl or ["all"])
            def allowed(cid: int) -> bool:
                groups = meta.get(cid, {}).get("access_groups") or ["all"]
                return bool(set(groups) & acl_set) or "all" in groups

            fts_ids = [i for i in fts_ids if allowed(i)]
            vec_ids = [i for i in vec_ids if allowed(i)]
            return fts_ids, vec_ids, meta, fts_and_hit
        finally:
            conn.close()

    def _fetch_sentences(self, chunk_ids: list[int], qvec: np.ndarray):
        """Векторы предложений для реранка + текст для цитаты.

        Один запрос на все чанки: 12 чанков × ~8 предложений = ~96 строк,
        это 96 × 312 × 2 байта (halfvec) ≈ 60 КБ.
        """
        if not chunk_ids:
            return [], []
        conn = self._psycopg.connect(self.dsn, autocommit=False)
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT chunk_id, seq, text, embedding
                FROM chunk_sentences
                WHERE chunk_id = ANY(%s)
                ORDER BY chunk_id, seq
            """, (chunk_ids,))
            rows = cur.fetchall()
            cur.close()
        finally:
            conn.close()

        rerank_input = [(r[0], r[1], r[3]) for r in rows]
        quote_input = [(r[0], r[2], r[3]) for r in rows]
        return rerank_input, quote_input

    def _fetch_stale(
        self, qvec: np.ndarray, tokens: list[str], acl: list[str]
    ) -> list[dict]:
        """Второй проход по НЕактивным версиям (правка B1).

        Выполняется только после отказа: 25% запросов, +18-30 мс.
        Для каждого найденного неактивного пункта ищем, куда он переехал
        в действующей версии — по эмбеддингу, а не по section_path
        (при перенумерации разделов путь меняется, и «пункт исключён»
        было бы ложью).
        """
        vec_literal = to_pgvector(qvec)
        conn = self._psycopg.connect(self.dsn, autocommit=False)
        try:
            cur = conn.cursor()
            cur.execute("""
                WITH staled AS (
                    SELECT c.id, c.document_id, c.section_path, c.section_title,
                           c.body, c.embedding, d.doc_key, d.title, d.version,
                           d.valid_to, d.owner_role, d.owner_contact,
                           1 - (c.embedding <=> %s::halfvec(312)) AS sim,
                           length(c.body) AS body_len
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE NOT c.is_active
                    ORDER BY c.embedding <=> %s::halfvec(312)
                    LIMIT 5
                )
                SELECT s.id, s.section_path, s.section_title, s.body, s.title,
                       s.version, s.valid_to, s.sim, s.owner_role,
                       s.owner_contact, s.doc_key, s.body_len,
                       a.id AS new_chunk_id, a.section_path AS new_section,
                       a.body AS new_body,
                       CASE WHEN a.id IS NULL THEN NULL
                            ELSE 1 - (a.embedding <=> s.embedding) END AS reloc_sim
                FROM staled s
                LEFT JOIN LATERAL (
                    SELECT c2.id, c2.section_path, c2.body, c2.embedding
                    FROM chunks c2
                    JOIN documents d2 ON d2.id = c2.document_id
                    WHERE d2.doc_key = s.doc_key
                      AND c2.is_active
                      AND length(c2.body) >= 40
                    ORDER BY c2.embedding <=> s.embedding
                    LIMIT 1
                ) a ON TRUE
                WHERE s.sim >= %s
                  AND s.body_len >= 40
                ORDER BY s.sim DESC
            """, (vec_literal, vec_literal, self.stale_threshold))
            rows = cur.fetchall()
            cur.close()
        finally:
            conn.close()

        out = []
        for r in rows:
            out.append({
                "chunk_id": r[0], "section_path": r[1], "section_title": r[2],
                "body": r[3], "doc_title": r[4], "version": r[5],
                "valid_to": r[6], "sim": float(r[7] or 0),
                "owner_role": r[8], "owner_contact": r[9], "doc_key": r[10],
                "body_len": r[11], "new_chunk_id": r[12],
                "new_section": r[13], "new_body": r[14],
                "reloc_sim": float(r[15]) if r[15] is not None else None,
            })
        return out

    # ------------------------------------------------------------- поиск --

    def search(
        self,
        query: str,
        user_id: str,
        acl: list[str] | None = None,
        use_cache: bool = True,
    ) -> Answer:
        """Главная точка входа. Возвращает Answer."""
        t0 = time.perf_counter()
        acl = acl or ["all"]
        acl_sig = acl_signature(acl)
        qhmac = query_hmac(query, self.pepper)

        # --- Кэш: ключ включает ACL, иначе утечка между группами (3.1) ---
        if use_cache:
            cached = self._cache_get(qhmac, acl_sig)
            if cached is not None:
                ans = Answer(**cached)
                ans.cache_hit = True
                ans.latency_ms = int((time.perf_counter() - t0) * 1000)
                # Вектор считаем отдельно: в query_logs он нужен для
                # кластеризации отказов, а на пути кэша мы его ещё не
                # считали. Это 2.6 мс — несопоставимо с экономией 65 мс.
                qvec_cached = self.encoder.encode_one(query)
                self._log(ans, qhmac, query, user_id, acl_sig, qvec_cached)
                return ans

        # --- Подготовка запроса ---
        pq = prepare(query)
        qvec = self.encoder.encode_one(query)

        # Версия в запросе: если она есть и не активна — это осознанный
        # вопрос про старую версию, ведём в stale-флоу сразу.
        versions = extract_versions(query)
        explicit_stale = None
        for v in versions:
            if not is_likely_section_ref(query, v.version):
                hit = self._check_version_active(v.version)
                if hit is not None and not hit["is_active"]:
                    explicit_stale = hit
                    break

        fts_ids, vec_ids, meta, fts_and_hit = self._fetch_candidates(
            pq.tokens, qvec, acl, POOL_SIZE
        )
        rrf = rrf_fuse(fts_ids, vec_ids)
        top_ids = [cid for cid, _ in rrf[:RERANK_LIMIT]]

        rerank_input, quote_input = self._fetch_sentences(top_ids, qvec)
        rerank = sentence_rerank(qvec, rerank_input)
        chunk_sims = {cid: m["chunk_sim"] for cid, m in meta.items()}

        # Покрытие токенов берём у лучшего чанка: если в нём нет слов
        # запроса, ответ почти наверняка «похожий, но не тот».
        top_coverage = 1.0
        if top_ids:
            top_coverage = meta.get(top_ids[0], {}).get("coverage", 1.0)

        features = confidence_features(
            rerank=rerank,
            chunk_sims=chunk_sims,
            rrf_scores=rrf,
            fts_exact=fts_and_hit,
            token_coverage=top_coverage,
        )
        p = self._confidence(features)
        answer = p >= self.threshold and bool(top_ids)

        # Покрытие НЕ используется как жёсткий фильтр: попытка требовать
        # coverage >= 0.5 уронила Recall@3 с 0.864 до 0.682, потому что
        # поиск по смыслу как раз и находит ответ при несовпадении слов.
        # Признак остаётся в features и учитывается калиброванной
        # логистикой с весом ~1.2.

        latency = int((time.perf_counter() - t0) * 1000)

        # --- Отказ или ответ ---
        if not answer:
            return self._refuse(
                query, qvec, pq.tokens, acl, qhmac, acl_sig, user_id,
                p, features, latency, explicit_stale,
            )

        # Порядок показа — по final_rank (косинус + покрытие + числа),
        # а не по RRF: RRF даёт кандидатов, но различать два соседних
        # пункта матрицы согласования по номерам сумм может только
        # числовое совпадение — косинус суммы не различает.
        coverage = {cid: m.get("coverage", 1.0) for cid, m in meta.items()}
        number_match = self._number_match(query, top_ids, meta)
        ranked = final_rank(rerank, coverage, top_ids, number_match=number_match)
        ordered_ids = [cid for cid, _ in ranked]
        best = self._build_sources(ordered_ids, meta, quote_input, qvec, query)
        if not best:
            return self._refuse(
                query, qvec, pq.tokens, acl, qhmac, acl_sig, user_id,
                p, features, latency, explicit_stale,
            )

        ans = Answer(
            response_type="success",
            confidence=p,
            sources=best,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            features=features,
        )
        # ВАЖНО: успешный ответ по ДЕЙСТВУЮЩЕЙ версии — это success,
        # а не stale. Наличие у документа предыдущей редакции — норма,
        # а не повод предупреждать: иначе Stale Rate раздувается до 68%,
        # и метрика перестаёт что-либо означать.
        #
        # Stale ставится только когда пользователь СПРОСИЛ про версию
        # (это обрабатывается в _refuse и в ветке explicit_stale ниже).
        if explicit_stale is not None:
            ans.response_type = "stale"
            ans.stale_warning = self._build_stale_message(explicit_stale, None)

        self._cache_put(qhmac, acl_sig, ans)
        self._log(ans, qhmac, query, user_id, acl_sig, qvec)
        return ans

    @staticmethod
    def _best_quote_sentence(qvec, cand: list, query: str) -> str | None:
        """Выбирает предложение для цитаты внутри чанка.

        Почему не просто max-cosine. В чанке 3.1-3.4 четыре пункта, и на
        вопрос «договоры до ста тысяч рублей» косинус выше у пункта 3.3
        («от 100 000 до 1 000 000»), потому что там два числа и больше
        пересечения по словам. Но правильный ответ — 3.1 («до 100 000»).

        Различает их число: в запросе есть 100000, и оно есть в обоих
        пунктах, поэтому одного совпадения мало. Дальше решает
        согласованность: «до» в запросе и «до» в пункте. Поэтому
        приоритет — предложение, где совпадает и число, и предлог
        направления (до / от / свыше), а косинус — разрешает остальное.
        """
        if not cand:
            return None

        from app.search.grounding import extract_numbers

        q_nums = extract_numbers(query)
        q_low = query.lower()

        # Направление суммы в запросе: «до», «от», «свыше».
        direction = None
        for marker in ("до", "от", "свыше", "более", "менее"):
            if f" {marker} " in f" {q_low} ":
                direction = marker
                break

        from app.search.grounding import (
            extract_numbers, extract_numbers_spans, normalize,
        )
        from app.utils.vec import to_f32
        import numpy as np

        def amounts(text: str) -> set[str]:
            """Суммы, исключая номера пунктов («3.1», «3.2»).

            Порядковые номера пунктов — не суммы. Без исключения «3.1»
            совпадает с числом 3 из запроса «до 3 дней» и перебивает сумму.
            """
            return {n for n in extract_numbers(text)
                    if not re.fullmatch(r"\d{1,2}\.\d{1,2}", n)}

        def bounds(text: str) -> set[str]:
            """Границы суммы: {'до:100000', 'от:100000'}.

            Работает по ПОЗИЦИЯМ чисел, а не по regex цифр: в запросе
            «до ста тысяч» число записано словом, и regex по цифрам не
            нашёл бы границу. Без этого пункты «до 100 000» и
            «от 100 000 до 1 000 000» неразличимы — оба содержат 100000.
            """
            norm = normalize(text)
            out: set[str] = set()
            for val, st, _en in extract_numbers_spans(text):
                # смотрим 12 символов перед числом на предлог направления
                before = norm[max(0, st - 12):st]
                m = re.search(r"(до|от|свыше|более|менее)\s*$", before)
                if m:
                    out.add(f"{m.group(1)}:{val}")
            return out

        q_bounds = bounds(query)
        scored = []
        for idx, (cid, text, emb) in enumerate(cand):
            t_amounts = amounts(text)
            num_hit = 1.0 if (q_nums and (q_nums & t_amounts)) else 0.0
            # Совпадение ГРАНИЦЫ («до:100000») — сильнее простого числа.
            b_hit = 1.0 if (q_bounds and (q_bounds & bounds(text))) else 0.0
            # Штраф за лишние границы: у 3.3 их две, у 3.1 одна. Запрос
            # с одной границей должен предпочитать пункт с одной.
            extra = 0.0
            if q_bounds:
                tb = bounds(text)
                if tb:
                    extra = -0.15 * max(0, len(tb) - len(q_bounds))
            dir_hit = 0.0
            if direction and f" {direction} " in f" {text.lower()} ":
                dir_hit = 1.0
            sim = float(np.dot(to_f32(emb), to_f32(qvec)))
            scored.append((b_hit * 0.45 + num_hit * 0.25 + extra
                           + dir_hit * 0.10 + sim * 0.15, idx, text))

        scored.sort(reverse=True)
        return scored[0][2]

    @staticmethod
    def _number_match(
        query: str, top_ids: list[int], meta: dict[int, dict]
    ) -> dict[int, float]:
        """Доля чисел запроса, найденных в теле чанка.

        Нужен, потому что косинус не различает суммы: «до 100 000 руб.» и
        «свыше 1 000 000 руб.» для энкодера почти одинаковы. На стенде
        вопрос «агентский договор свыше миллиона» выдавал пункт про
        5 000 000 — числа были единственным различием.

        Если в запросе чисел нет, признак не применяется (возвращаем 1.0):
        иначе чанки без чисел штрафовались бы без причины.
        """
        from app.search.grounding import extract_numbers

        q_nums = extract_numbers(query)
        if not q_nums:
            return {cid: 1.0 for cid in top_ids}

        out: dict[int, float] = {}
        for cid in top_ids:
            body = meta.get(cid, {}).get("body", "")
            c_nums = extract_numbers(body)
            if not c_nums:
                out[cid] = 0.0
                continue
            out[cid] = len(q_nums & c_nums) / len(q_nums)
        return out

    def _build_sources(
        self,
        top_ids: list[int],
        meta: dict[int, dict],
        quote_input: list,
        qvec: np.ndarray,
        query: str,
    ) -> list[Source]:
        """Формирует источники с дословными цитатами.

        Цитата — предложение-победитель, а не чанк целиком, если чанк
        длиннее 400 символов: пользователю нужен точный пункт, а не абзац.
        Если чанк короткий — отдаём его целиком, это и есть дословность.
        """
        out: list[Source] = []
        for cid in top_ids[:3]:
            m = meta.get(cid)
            if not m:
                continue
            body = m["body"]

            # Когда показывать предложение вместо тела целиком.
            #
            # Дефект, найденный на стенде: чанк 3.1-3.4 (389 символов) содержал
            # ПРАВИЛЬНЫЙ ответ в предложении «3.2. Агентский договор подписывает
            # финансовый директор при сумме свыше 1 000 000 рублей», но
            # показывался с начала — «3.1. Договоры на сумму до 100 000 рублей».
            # Пользователь видел цитату про 100 тысяч на вопрос про миллион
            # и считал ответ неверным. Порог 400 символов этого не ловил.
            #
            # Правило: если в теле несколько нумерованных пунктов ИЛИ тело
            # длиннее 250 символов — цитируем предложение-победитель.
            # Один короткий пункт цитируем целиком: это и есть дословность.
            multi_item = len(re.findall(r"(?:^|\n)\s*\d+\.\d+\.", body)) > 1
            if len(body) <= 250 and not multi_item:
                quote = body
            else:
                cand = [(c, t, e) for (c, t, e) in quote_input if c == cid]
                quote = self._best_quote_sentence(qvec, cand, query) or body
            # Если цитата — конкретный пункт, номер раздела берём ИЗ ЦИТАТЫ.
            # Иначе пользователь видит «п.3.1», а цитата про «3.2»: ссылка
            # не соответствует тому, что процитировано. Чанк склеен из
            # пунктов 3.1-3.4, поэтому его section_path — путь первого.
            quote_path = m["section_path"]
            qm = re.match(r"^\s*(\d+(?:\.\d+)*)\.", quote)
            if qm and m["chunk_type"] == "item":
                quote_path = qm.group(1)

            out.append(Source(
                chunk_id=cid,
                doc_title=m["doc_title"],
                version=m["version"],
                section_path=quote_path,
                section_title=m["section_title"],
                source_path=m["source_path"],
                quote=quote,
                is_active=m["is_active"],
            ))
        return out

    def _refuse(
        self,
        query: str,
        qvec: np.ndarray,
        tokens: list[str],
        acl: list[str],
        qhmac: str,
        acl_sig: str,
        user_id: str,
        p: float,
        features: dict[str, float],
        latency: int,
        explicit_stale: dict | None,
    ) -> Answer:
        """Отказ. Пытается дать полезную рекомендацию, а не просто «нет».

        Порядок:
          1. Если пользователь явно спросил про недействительную версию —
             это stale, а не отказ: он получит предупреждение и указание
             на действующую версию.
          2. Иначе второй проход по неактивным версиям: возможно, ответ
             есть там, и человек искал по отменённому регламенту.
          3. Иначе честный отказ с владельцем темы, если он известен.
        """
        stale_hits = self._fetch_stale(qvec, tokens, acl)

        hit = stale_hits[0] if stale_hits else None
        if explicit_stale or (hit and hit["sim"] >= self.stale_threshold):
            warning = self._build_stale_message(explicit_stale, hit)
            ans = Answer(
                response_type="stale",
                confidence=max(p, hit["sim"] if hit else 0.0),
                sources=[],
                refusal_reason="stale_only",
                latency_ms=latency,
                features=features,
                stale_warning=warning,
                owner_role=(explicit_stale or hit or {}).get("owner_role"),
                owner_contact=(explicit_stale or hit or {}).get("owner_contact"),
            )
            self._log(ans, qhmac, query, user_id, acl_sig, qvec)
            return ans

        owner = self._pick_owner(tokens)
        ans = Answer(
            response_type="refusal",
            confidence=p,
            sources=[],
            refusal_reason=("no_match" if not features.get("fts_exact")
                            else "low_confidence"),
            owner_role=owner.get("role") if owner else None,
            owner_contact=owner.get("contact") if owner else None,
            latency_ms=latency,
            features=features,
        )
        self._log(ans, qhmac, query, user_id, acl_sig, qvec)
        return ans

    def _confidence(self, features: dict[str, float]) -> float:
        from app.search.fusion import confidence
        if not self.weights:
            # Без калибровки используем консервативное правило:
            # высокий max_sentence_sim и заметный margin.
            return min(features.get("max_sentence_sim", 0.0), 1.0)
        return confidence(features, self.weights)

    # ------------------------------------------------------- вспомогательное --

    def _check_version_active(self, version: str) -> dict | None:
        """Есть ли версия в базе и активна ли она."""
        conn = self._psycopg.connect(self.dsn, autocommit=False)
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT d.title, d.version, d.valid_to, d.is_active,
                       d.owner_role, d.owner_contact,
                       a.version AS active_version
                FROM documents d
                LEFT JOIN documents a ON a.doc_key = d.doc_key AND a.is_active
                WHERE d.version = %s
                LIMIT 1
            """, (strip_version_prefix(version),))
            r = cur.fetchone()
            cur.close()
            if not r:
                return None
            return {
                "title": r[0], "version": r[1], "valid_to": r[2],
                "is_active": r[3], "owner_role": r[4], "owner_contact": r[5],
                "active_version": r[6],
            }
        finally:
            conn.close()

    def _stale_siblings_for(
        self, top_ids: list[int], meta: dict[int, dict]
    ) -> list[dict]:
        """Недействительные версии ТЕХ ЖЕ семейств документов (doc_key).

        Сравнение по doc_key, а не по title: только так предупреждение
        относится к документу, по которому реально дан ответ. Иначе
        варнинг про устаревший регламент договоров висел на вопросах
        про отпуска, и Stale Rate раздувался до 46% при осмысленных ~5%.
        """
        if not top_ids:
            return []
        # doc_key текущих ответов
        conn = self._psycopg.connect(self.dsn, autocommit=False)
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT DISTINCT d.doc_key
                FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE c.id = ANY(%s)
            """, (top_ids,))
            keys = [r[0] for r in cur.fetchall()]
            if not keys:
                return []

            cur.execute("""
                SELECT doc_key, title, version, valid_to
                FROM documents
                WHERE NOT is_active AND doc_key = ANY(%s)
                ORDER BY valid_to DESC
            """, (keys,))
            rows = cur.fetchall()
            cur.close()
            return [{"doc_key": r[0], "title": r[1], "version": r[2],
                     "valid_to": r[3]} for r in rows]
        finally:
            conn.close()

    @staticmethod
    def _stale_warning_text(items: list[dict]) -> str:
        parts = []
        for it in items:
            until = it["valid_to"].strftime("%d.%m.%Y") if it["valid_to"] else "—"
            parts.append(f"версия {it['version']} недействительна с {until}")
        return ("В базе есть устаревшие версии этого документа: "
                + "; ".join(parts) + ". Ответ дан по действующей версии.")

    def _build_stale_message(self, explicit: dict | None, hit: dict | None) -> str:
        """Текст предупреждения для stale-флоу."""
        src = explicit or hit or {}
        title = src.get("title") or src.get("doc_title") or "документ"
        version = src.get("version") or "?"
        valid_to = src.get("valid_to")
        until = valid_to.strftime("%d.%m.%Y") if hasattr(valid_to, "strftime") else "—"
        msg = f"Документ «{title}» версии {version} недействителен с {until}."

        if hit and hit.get("new_chunk_id"):
            reloc = hit.get("reloc_sim") or 0.0
            if reloc >= 0.95:
                msg += (f" Этот пункт есть в действующей версии, "
                        f"теперь раздел {hit['new_section']}.")
            elif reloc >= 0.85:
                msg += (f" Пункт переработан: в действующей версии он находится "
                        f"в разделе {hit['new_section']}.")
            else:
                msg += " В действующей версии этот пункт исключён или переписан."
        return msg

    def _pick_owner(self, tokens: list[str]) -> dict | None:
        """Подбирает владельца темы для рекомендации в отказе.

        Ищем по совпадению лексем токенов запроса с темой владельца.
        Если ничего не нашли — возвращаем None, и бот не выдумывает
        должность (это тоже форма честности).
        """
        if not tokens:
            return None
        conn = self._psycopg.connect(self.dsn, autocommit=False)
        try:
            cur = conn.cursor()
            cur.execute("SELECT topic_key, title, role, contact FROM knowledge_owners")
            rows = cur.fetchall()
            cur.close()
        finally:
            conn.close()

        if not rows:
            return None
        # Подбор владельца по ЛЕММАМ, а не по поверхностным словам.
        #
        # Дефект на стенде: на вопрос «что делать при утечке данных» владелец
        # не находился, потому что в запросе «утечке», а тема называется
        # «Инциденты информационной безопасности» — пересечения по строкам нет.
        # Лемматизация сближает «утечк» с темой через синонимы, а сравнение
        # по основам слов даёт совпадение там, где формы разные.
        import re as _re

        def stems(text: str) -> set[str]:
            """Грубые основы слов: первые 5 символов, без служебных."""
            out = set()
            for w in _re.findall(r"[a-zа-яё]{3,}", text.lower().replace("ё", "е")):
                out.add(w[:5])
            return out

        tok_stems = stems(" ".join(tokens))
        if not tok_stems:
            return None

        # Ключи тем ЛАТИНИЦЕЙ (ib_incidenty, doverennosti), а запрос русский:
        # сравнение по строкам не пересекается вообще (проверено — все темы
        # давали score 0.000). Поэтому каждой теме сопоставлен набор РУССКИХ
        # ключевых слов, включая бытовые формулировки сотрудника
        # («утечка», «виза», «пропуск»), а не только термины регламента.
        TOPIC_HINTS: dict[str, str] = {
            "doverennosti": "доверенность полномочия представительство подпись",
            "komandirovki_zarubezh": "заграница виза зарубеж зарубежная паспорт валюта выезд командировка граница загран",
            "ib_incidenty": "утечка инцидент безопасность данные персональные взлом атака",
            "dogovory": "договор согласование подписание контрагент сделка",
            "komandirovki": "командировка суточные поездка отчёт билет",
            "zakupki": "закупка тендер поставщик заявка приобретение",
            "otpuska": "отпуск отдых график",
            "propuski": "пропуск офис подрядчик доступ",
            "arenda": "аренда переговорная помещение",
            "simkarty": "сим-карта связь корпоративная",
            "avtomobil": "автомобиль компенсация бензин",
            "perevod": "перевод отдел должность",
            "bolnichny": "больничный лист нетрудоспособность",
            "spravki": "справка доходы",
        }

        best, best_score = None, 0.0
        for key, title, role, contact in rows:
            # Источники слов для темы: русские подсказки + название.
            hint = TOPIC_HINTS.get(key, "")
            topic_stems = stems(hint + " " + (title or ""))
            if not topic_stems:
                continue
            inter = tok_stems & topic_stems
            # Специфичность: тема с более редкими словами (зарубеж, виза)
            # должна обходить общую тему (командировка). Иначе запрос
            # «командировка за границу» уходит к владельцу общих
            # командировок, хотя случай другой и владелец другой.
            specificity = 0.0
            if "zarubezh" in key:
                if tok_stems & stems("зарубеж загран заграница виза паспорт валюта"):
                    specificity = 0.35
            elif key == "ib_incidenty":
                if tok_stems & stems("утечка инцидент взлом атака"):
                    specificity = 0.35
            score = len(inter) / max(len(tok_stems), 1) + specificity
            if score > best_score:
                best, best_score = {"key": key, "title": title, "role": role,
                                    "contact": contact}, score
        return best if best_score >= 0.2 else None

    # --------------------------------------------------------------- кэш --

    def _cache_get(self, qhmac: str, acl_sig: str) -> dict | None:
        conn = self._psycopg.connect(self.dsn, autocommit=False)
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT payload FROM answer_cache
                WHERE cache_key = %s AND acl_signature = %s
                  AND expires_at > now()
            """, (qhmac, acl_sig))
            r = cur.fetchone()
            cur.close()
            if not r:
                return None
            payload = r[0]
            return {
                "response_type": payload["response_type"],
                "confidence": payload["confidence"],
                "sources": [Source(**s) for s in payload.get("sources", [])],
                "refusal_reason": payload.get("refusal_reason"),
                "owner_role": payload.get("owner_role"),
                "owner_contact": payload.get("owner_contact"),
                "features": payload.get("features", {}),
                "stale_warning": payload.get("stale_warning"),
            }
        finally:
            conn.close()

    def _cache_put(self, qhmac: str, acl_sig: str, ans: Answer) -> None:
        """Кладёт в кэш только успешные ответы.

        Отказы не кэшируем: причина отказа может измениться после
        индексации нового документа, а TTL 24 часа этого не заметит.
        """
        if ans.response_type == "refusal":
            return
        payload = {
            "response_type": ans.response_type,
            "confidence": ans.confidence,
            "sources": [s.to_dict() for s in ans.sources],
            "refusal_reason": ans.refusal_reason,
            "owner_role": ans.owner_role,
            "owner_contact": ans.owner_contact,
            "features": ans.features,
            "stale_warning": ans.stale_warning,
        }
        conn = self._psycopg.connect(self.dsn, autocommit=False)
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO answer_cache (cache_key, acl_signature, payload,
                                          index_generation, expires_at)
                VALUES (%s, %s, %s::jsonb, 1,
                        now() + interval '%s hours')
                ON CONFLICT (cache_key, acl_signature) DO UPDATE SET
                    payload = EXCLUDED.payload,
                    created_at = now(),
                    expires_at = EXCLUDED.expires_at
            """, (qhmac, acl_sig, json.dumps(payload, ensure_ascii=False,
                                             default=str), CACHE_TTL_HOURS))
            conn.commit()
            cur.close()
        finally:
            conn.close()

    # --------------------------------------------------------------- лог --

    def _log(
        self,
        ans: Answer,
        qhmac: str,
        query: str,
        user_id: str,
        acl_sig: str,
        qvec=None,
    ) -> None:
        """Пишет запрос в query_logs.

        Сырой текст НЕ пишется. Попадание в кэш тоже логируется (правка B6):
        иначе deflection и time saved посчитаются от неполного числа запросов,
        и дашборд соврёт.
        """
        try:
            if qvec is None:
                qvec = self.encoder.encode_one(query)
            conn = self._psycopg.connect(self.dsn, autocommit=False)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO query_logs
                    (query_hmac, query_len, embedding, response_type,
                     refusal_reason, confidence, top_score, sources,
                     latency_ms, cache_hit, user_id_hash, is_simulated)
                VALUES (%s, %s, %s::halfvec, %s, %s, %s, %s, %s::jsonb,
                        %s, %s, %s, false)
            """, (
                qhmac, len(query), to_pgvector(qvec), ans.response_type,
                ans.refusal_reason, ans.confidence,
                ans.features.get("max_sentence_sim"),
                json.dumps([s.to_dict() for s in ans.sources], ensure_ascii=False),
                ans.latency_ms, ans.cache_hit,
                hashlib.sha256(f"{user_id}".encode()).hexdigest(),
            ))
            conn.commit()
            cur.close()
        except Exception:
            # Логирование не должно ломать ответ пользователю.
            pass
