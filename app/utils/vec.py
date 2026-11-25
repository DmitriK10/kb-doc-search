"""Единая точка конвертации векторов в float32.

Правка К5: pgvector-драйвер (psycopg + pgvector) для halfvec возвращает
объект Halfvec, а для vector — numpy-массив. np.asarray(...) на Halfvec
падает или даёт мусор. Поэтому все места, где вектор идёт в numpy,
обязаны проходить через to_f32().

Правило: конверсия живёт здесь, а не в девяти вызовах по коду.
Потребление RAM: float32-копия одного вектора 312 * 4 = 1248 байт.
"""

from __future__ import annotations

import numpy as np

# Размерность rubert-tiny2. Для intfloat/multilingual-e5-small — 384.
EMB_DIM = 312


def to_f32(vec) -> np.ndarray:
    """Приводит любой векторовидный объект к np.ndarray[float32].

    Поддерживает:
      - np.ndarray
      - pgvector.Halfvec (есть .to_numpy())
      - pgvector.Vector / список / кортеж (итерируемое)
      - строку вида '[0.1,0.2,...]' (текстовое представление из SQL)

    float16 -> float32 обязателен: halfvec хранит 16 бит, накопление
    в 16 битах даёт заметную ошибку косинуса на 312 измерениях.
    """
    if isinstance(vec, np.ndarray):
        return vec.astype(np.float32, copy=False)

    # pgvector >= 0.2: Halfvec и Vector реализуют to_numpy()
    if hasattr(vec, "to_numpy"):
        return np.asarray(vec.to_numpy(), dtype=np.float32)

    if isinstance(vec, (str, bytes)):
        return parse_pgvector_text(vec)

    return np.asarray(list(vec), dtype=np.float32)


def parse_pgvector_text(raw) -> np.ndarray:
    """Парсит '[1,2,3]' или 'halfvec:[1,2,3]' из текстового вывода."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    body = raw.strip()
    if ":" in body and not body.startswith("["):
        # 'halfvec:[0.1,...]' -> '[0.1,...]'
        body = body.split(":", 1)[1]
    body = body.strip().lstrip("[").rstrip("]")
    if not body:
        return np.zeros(EMB_DIM, dtype=np.float32)
    return np.fromstring(body, sep=",", dtype=np.float32)


def to_matrix(rows, dim: int = EMB_DIM) -> np.ndarray:
    """Собирает матрицу (n, dim) из списка векторов. Пустой вход -> (0, dim)."""
    if not rows:
        return np.zeros((0, dim), dtype=np.float32)
    flat = [to_f32(r) for r in rows]
    return np.vstack(flat)


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    """L2-нормализация по строкам. Защита от нулевых векторов — в знаменателе."""
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-9)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-нормализация одного вектора."""
    arr = np.asarray(vec, dtype=np.float32)
    return arr / max(float(np.linalg.norm(arr)), 1e-9)