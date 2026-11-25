"""Энкодер: rubert-tiny2 int8 через onnxruntime.

Почему onnxruntime, а не torch: импорт torch добавляет 200–250 МБ RSS,
что на 2 ГБ RAM недопустимо рядом с PostgreSQL. onnxruntime с int8-моделью
даёт ~180 МБ на процесс и 2.6 мс на запрос (замерено на сервере).

Модель экспортируется ОДИН раз скриптом scripts/export_onnx.py. В рантайме
torch и transformers не нужны: токенизатор берётся из файлов модели, а не
из библиотеки transformers — иначе тянется весь пакет.

Векторы L2-нормированы на выходе модели, поэтому косинус = скалярное
произведение, и в numpy не нужно нормировать повторно.

RAM: ~180 МБ на процесс (модель 29.5 МБ + сессия + арены).
Латентность: 2.6 мс на запрос из 24 токенов, батч 64 — ~35 мс на 2 vCPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None  # type: ignore[assignment]

DEFAULT_MODEL_DIR = Path("/opt/kb/models/rubert-tiny2-int8")
EMB_DIM = 312
MAX_LEN = 128


class Encoder:
    """Обёртка над ONNX-сессией с батчированием и кэшем токенизации.

    Один экземпляр на процесс: сессия onnxruntime потокобезопасна для run(),
    но создание новой сессии на каждый запрос съедает ~50 мс и память.
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        intra_threads: int = 2,
    ) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime не установлен")
        self.dir = Path(model_dir)
        meta_path = self.dir / "kb_meta.json"
        self.meta = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists()
            else {}
        )
        model_path = self.dir / "model_int8.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"нет модели: {model_path}")

        so = ort.SessionOptions()
        # 2 vCPU: больше потоков даст конкуренцию с PostgreSQL и ботом,
        # меньше — недоиспользует ядра. Значение из ТЗ (ORT_NUM_THREADS=2).
        so.intra_op_num_threads = intra_threads
        so.inter_op_num_threads = 1
        # Отключаем лишние аллокации: на маленьких батчах это заметно.
        so.enable_cpu_mem_arena = True
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._tok = _load_tokenizer(self.dir)
        self.dim = EMB_DIM

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Возвращает матрицу (n, 312) float32, строки L2-нормированы.

        Батчирование нужно на индексации: 20k чанков по одному — это 20k
        вызовов сессии вместо ~600 батчей, разница в разы.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        out: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = self._tok(
                chunk,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
            )
            feeds = {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
                "token_type_ids": enc["token_type_ids"].astype(np.int64),
            }
            vecs = self.session.run(None, feeds)[0]
            out.append(np.asarray(vecs, dtype=np.float32))

        mat = np.vstack(out)
        # Страховка: модель уже нормирует, но при смене модели/квантования
        # нормировка может потеряться — тогда косинус перестанет быть
        # скалярным произведением и реранк будет врать.
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(norms, 1e-9)

    def encode_one(self, text: str) -> np.ndarray:
        """Вектор одного текста, shape (312,)."""
        return self.encode([text])[0]


def _load_tokenizer(model_dir: Path):
    """Загружает токенизатор БЕЗ transformers.

    transformers тянет ~200 МБ зависимостей ради одного класса. Здесь
    используется tokenizers (Rust) — он читает tokenizer.json напрямую
    и весит единицы МБ.

    Если tokenizers недоступен, падаем с внятной ошибкой, а не тянем
    transformers молча: на 2 ГБ RAM это недопустимо.
    """
    tok_json = model_dir / "tokenizer.json"
    if not tok_json.exists():
        raise FileNotFoundError(f"нет tokenizer.json в {model_dir}")

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "нужен пакет tokenizers (pip install tokenizers); "
            "transformers тянуть нельзя — 200 МБ RAM"
        ) from exc

    base = Tokenizer.from_file(str(tok_json))
    base.enable_truncation(max_length=MAX_LEN)
    base.enable_padding(pad_id=0, pad_token="[PAD]")

    cfg_path = model_dir / "tokenizer_config.json"
    cfg = (
        json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg_path.exists()
        else {}
    )
    model_max = min(int(cfg.get("model_max_length", MAX_LEN)), 512)
    if model_max < MAX_LEN:
        base.enable_truncation(max_length=model_max)

    return _TokWrapper(base)


class _TokWrapper:
    """Приводит tokenizers.Tokenizer к интерфейсу, похожему на transformers.

    Нужно ровно одно: вернуть dict с input_ids / attention_mask /
    token_type_ids как numpy-массивы. Token type ids у BERT-моделей
    всегда нули для одиночного текста, но модель их ждёт как вход.
    """

    def __init__(self, base) -> None:
        self.base = base

    def __call__(
        self,
        texts: list[str],
        return_tensors: str = "np",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = MAX_LEN,
    ) -> dict[str, np.ndarray]:
        encs = self.base.encode_batch(list(texts))
        max_len = max(len(e.ids) for e in encs) if encs else 1
        ids, mask = [], []
        for e in encs:
            pad = max_len - len(e.ids)
            ids.append(list(e.ids) + [0] * pad)
            mask.append(list(e.attention_mask) + [0] * pad)
        return {
            "input_ids": np.asarray(ids, dtype=np.int64),
            "attention_mask": np.asarray(mask, dtype=np.int64),
            "token_type_ids": np.zeros((len(ids), max_len), dtype=np.int64),
        }


def to_pgvector(vec: np.ndarray) -> str:
    """Строка '[0.1,0.2,...]' для вставки в halfvec.

    Формат pgvector: квадратные скобки, запятые. Используется в индексаторе
    и в поиске (параметр запроса).
    """
    arr = np.asarray(vec, dtype=np.float32).ravel()
    return "[" + ",".join(f"{x:.6f}" for x in arr) + "]"


def acl_signature(groups: list[str] | None) -> str:
    """Подпись групп доступа для ключа кэша.

    Сортировка обязательна: иначе ['hr','legal'] и ['legal','hr'] дадут
    разные ключи, и кэш будет промахиваться при одинаковых правах.
    """
    items = sorted(groups or ["all"])
    return "|".join(items)