"""Экспорт rubert-tiny2 в ONNX int8 — ОДНОРАЗОВЫЙ скрипт, не часть рантайма.

Зачем экспорт обязателен: в репозитории cointegrated/rubert-tiny2 на HuggingFace
НЕТ готовых ONNX-файлов (только model.safetensors). Проверено запросом к API
модели. Поэтому int8-ONNX приходится собирать самому.

Почему int8, а не fp32: fp32 ONNX ~120 МБ, int8 ~30 МБ. На сервере с 2 ГБ RAM
и без дискретной GPU разница важна и по памяти, и по скорости на CPU.

Почему mean-pooling, а не CLS: у rubert-tiny2 нет обученного пулера
(в sentence_bert_config.json — mean pooling). Взять CLS-токен было бы
неверно: это не тот вектор, на котором модель обучалась.

Почему нормируем вектор на выходе: косинус тогда равен скалярному
произведению, и реранк в numpy сводится к одному matmul без повторной
нормализации на каждом запросе.

torch и transformers нужны ТОЛЬКО здесь. В прод едет onnxruntime.
Пиковое потребление RAM при экспорте: ~1.2 ГБ (torch + веса fp32).
Поэтому скрипт запускается на сервере ОДИН раз и не пересекается с API.

Запуск:
    python scripts/export_onnx.py --out /opt/kb/models/rubert-tiny2-int8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Экспорт rubert-tiny2 в ONNX int8")
    ap.add_argument("--model", default="cointegrated/rubert-tiny2")
    ap.add_argument("--out", default="/opt/kb/models/rubert-tiny2-int8")
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"1/5 загрузка токенизатора {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.save_pretrained(out)

    print("2/5 загрузка модели (fp32)")
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    hidden = model.config.hidden_size
    print(f"    hidden_size = {hidden}")

    # Обёртка с mean-pooling и L2-нормализацией.
    # Внимание: маска должна расширяться так же, как в transformers
    # (unsqueeze(-1) -> float), иначе сумма разойдётся с эталоном.
    class Encoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            out = self.m(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            last = out.last_hidden_state                      # (B, T, H)
            mask = attention_mask.unsqueeze(-1).float()       # (B, T, 1)
            summed = (last * mask).sum(dim=1)                 # (B, H)
            counts = mask.sum(dim=1).clamp(min=1e-9)          # (B, 1)
            pooled = summed / counts
            return torch.nn.functional.normalize(pooled, p=2, dim=1)

    enc = Encoder(model).eval()

    print("3/5 экспорт в ONNX (fp32)")
    dummy = tok(
        ["проверка экспорта модели в формат onnx"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
    )
    fp32_path = out / "model_fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(
            enc,
            (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
            str(fp32_path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["sentence_embedding"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "token_type_ids": {0: "batch", 1: "seq"},
                "sentence_embedding": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"    fp32: {fp32_path.stat().st_size / 1e6:.1f} МБ")

    print("4/5 динамическое int8-квантование")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = out / "model_int8.onnx"
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    print(f"    int8: {int8_path.stat().st_size / 1e6:.1f} МБ")

    print("5/5 проверка: сравнение fp32 и int8 на реальных парах")
    import numpy as np
    import onnxruntime as ort

    def run(path: Path, texts: list[str]) -> np.ndarray:
        sess = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        batch = tok(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        feeds = {
            "input_ids": batch["input_ids"].astype(np.int64),
            "attention_mask": batch["attention_mask"].astype(np.int64),
            "token_type_ids": batch["token_type_ids"].astype(np.int64),
        }
        return sess.run(None, feeds)[0]

    pairs = [
        ("кто подписывает агентский договор",
         "Агентский договор подписывает финансовый директор при сумме свыше 1 000 000 рублей."),
        ("кто подписывает агентский договор",
         "Суточные выплачиваются в размере 700 рублей за каждый день командировки."),
        ("как оформить командировку",
         "Отчёт о командировке сдаётся в бухгалтерию не позднее трёх рабочих дней."),
        ("срок согласования договора",
         "Срок согласования — не позднее 5 рабочих дней с момента передачи проекта."),
    ]
    a = run(fp32_path, [p[0] for p in pairs])
    b = run(int8_path, [p[0] for p in pairs])
    a_pos = run(fp32_path, [p[1] for p in pairs])
    b_pos = run(int8_path, [p[1] for p in pairs])

    sim32 = np.sum(a * a_pos, axis=1)
    sim8 = np.sum(b * b_pos, axis=1)
    drift = np.abs(sim32 - sim8)

    print("    косинус fp32:", np.round(sim32, 4).tolist())
    print("    косинус int8:", np.round(sim8, 4).tolist())
    print(f"    max |Δ| = {drift.max():.5f}, mean |Δ| = {drift.mean():.5f}")

    # Требование из ADR: просадка качества не больше 2-3 п.п.
    # Здесь мерим не Recall (для него нужен индекс), а сдвиг косинуса.
    ok = drift.max() < 0.02
    print(f"    сдвиг в допуске (<0.02): {'ДА' if ok else 'НЕТ'}")

    # Замер латентности на 2 vCPU: 24 токена — типичный запрос.
    text = ["кто подписывает агентский договор"]
    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    batch = tok(text, return_tensors="np", padding=True, truncation=True,
                max_length=args.max_length)
    feeds = {
        "input_ids": batch["input_ids"].astype(np.int64),
        "attention_mask": batch["attention_mask"].astype(np.int64),
        "token_type_ids": batch["token_type_ids"].astype(np.int64),
    }
    for _ in range(3):
        sess.run(None, feeds)
    t0 = time.perf_counter()
    n = 30
    for _ in range(n):
        sess.run(None, feeds)
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"    латентность int8: {ms:.1f} мс/запрос")

    meta = {
        "model": args.model,
        "hidden_size": hidden,
        "max_length": args.max_length,
        "pooling": "mean",
        "normalized": True,
        "files": {
            "int8": int8_path.name,
            "fp32": fp32_path.name,
        },
        "int8_size_mb": round(int8_path.stat().st_size / 1e6, 1),
        "latency_ms": round(ms, 2),
        "cosine_drift_max": round(float(drift.max()), 5),
        "note": "cosine = dot product, потому что вектор L2-нормирован",
    }
    (out / "kb_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nготово: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())