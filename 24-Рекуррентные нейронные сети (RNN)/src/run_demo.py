#!/usr/bin/env python3
"""Учебная Elman RNN на NumPy: узнаём направление последовательности.

Запуск из любой директории: python /путь/к/src/run_demo.py
Никаких загрузок данных; результат воспроизводится с фиксированным seed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def sigmoid(z):
    # В этой небольшой модели logits умеренны; clip дополнительно защищает exp.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700.0, 700.0)))


def make_pairs(seed=24, train_pairs=256, test_pairs=128, length=12):
    """Сначала делим ID пар, затем помещаем оба направления в один split.

    У каждой пары ровно одинаковый набор значений: label=1 — возрастающий
    порядок, label=0 — обратный. Сдвиг затрудняет классификацию по последней
    точке, но разность последней и первой всегда решает эту учебную задачу.
    """
    rng = np.random.default_rng(seed)
    pairs = []
    for pair_id in range(train_pairs + test_pairs):
        raw = np.sort(rng.uniform(-1.0, 1.0, length))
        raw = 2.0 * (raw - raw[0]) / (raw[-1] - raw[0]) - 1.0
        center = rng.uniform(-2.0, 2.0)
        half_span = rng.uniform(0.4, 1.0)
        pairs.append((pair_id, center + half_span * raw))
    ids = rng.permutation(len(pairs))
    splits = {}
    for name, split_ids in (("train", ids[:train_pairs]), ("test", ids[train_pairs:])):
        xs, ys, pair_ids = [], [], []
        for pair_id in split_ids:
            ascending = pairs[int(pair_id)][1]
            xs.extend((ascending, ascending[::-1]))
            ys.extend((1, 0))
            pair_ids.extend((int(pair_id), int(pair_id)))
        order = rng.permutation(len(xs))
        splits[name] = {
            "x_raw": np.asarray(xs, dtype=np.float64)[order, :, None],
            "y": np.asarray(ys, dtype=np.float64)[order, None],
            "pair_ids": np.asarray(pair_ids, dtype=np.int64)[order],
        }
    # Единственные параметры стандартизации вычисляются по train.
    mean = float(np.mean(splits["train"]["x_raw"]))
    std = float(np.std(splits["train"]["x_raw"]))
    for split in splits.values():
        split["x"] = (split["x_raw"] - mean) / std
    assert not set(splits["train"]["pair_ids"]) & set(splits["test"]["pair_ids"])
    return splits, {"train_mean": mean, "train_std": std}


def init_params(seed=25, hidden=8):
    rng = np.random.default_rng(seed)
    # Ортогональная инициализация рекуррентной матрицы с масштабом 0.8.
    q, _ = np.linalg.qr(rng.normal(size=(hidden, hidden)))
    return {
        "Wx": rng.normal(0.0, 0.5, (1, hidden)),
        "Wh": 0.8 * q,
        "bh": np.zeros((1, hidden)),
        "Wy": rng.normal(0.0, 1.0 / np.sqrt(hidden), (hidden, 1)),
        "by": np.zeros((1, 1)),
    }


def forward(p, x):
    """h_t=tanh(x_t Wx + h_(t-1) Wh + bh); logit=h_T Wy+by."""
    states = [np.zeros((x.shape[0], p["Wh"].shape[0]))]
    for t in range(x.shape[1]):
        states.append(np.tanh(x[:, t, :] @ p["Wx"] + states[-1] @ p["Wh"] + p["bh"]))
    logits = states[-1] @ p["Wy"] + p["by"]
    return logits, states


def loss_and_grad(p, x, y):
    logits, states = forward(p, x)
    # BCE, записанная через logits, устойчива для больших |logit|.
    loss = float(np.mean(np.logaddexp(0.0, logits) - y * logits))
    dz = (sigmoid(logits) - y) / len(y)
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    grads["Wy"] = states[-1].T @ dz
    grads["by"] = dz.sum(axis=0, keepdims=True)
    dh = dz @ p["Wy"].T
    # BPTT по всей последовательности, без усечения и до clipping.
    for t in reversed(range(x.shape[1])):
        da = dh * (1.0 - states[t + 1] ** 2)
        grads["Wx"] += x[:, t, :].T @ da
        grads["Wh"] += states[t].T @ da
        grads["bh"] += da.sum(axis=0, keepdims=True)
        dh = da @ p["Wh"].T
    return loss, grads


def evaluate(p, split):
    logits, _ = forward(p, split["x"])
    loss = float(np.mean(np.logaddexp(0.0, logits) - split["y"] * logits))
    probs = sigmoid(logits)
    accuracy = float(np.mean((probs >= 0.5) == split["y"]))
    return {"loss": loss, "accuracy": accuracy}, probs


def check_gradients(p, x, y, epsilon=1e-6):
    _, analytical = loss_and_grad(p, x, y)
    rows = []
    for name, value in p.items():
        for index in np.ndindex(value.shape):
            original = value[index]
            value[index] = original + epsilon
            plus, _ = loss_and_grad(p, x, y)
            value[index] = original - epsilon
            minus, _ = loss_and_grad(p, x, y)
            value[index] = original
            numerical = (plus - minus) / (2.0 * epsilon)
            exact = float(analytical[name][index])
            absolute_error = abs(numerical - exact)
            relative_error = absolute_error / max(1e-8, abs(numerical) + abs(exact))
            rows.append({"parameter": name, "index": str(index), "analytical": exact,
                         "finite_difference": float(numerical),
                         "absolute_error": float(absolute_error),
                         "relative_error": float(relative_error)})
    # Абсолютный критерий остаётся информативным для почти нулевых градиентов.
    passed = all(r["absolute_error"] < 1e-7 or r["relative_error"] < 1e-5 for r in rows)
    summary = {"passed": passed, "checked_parameters": len(rows), "epsilon": epsilon,
               "max_absolute_error": max(r["absolute_error"] for r in rows),
               "max_relative_error": max(r["relative_error"] for r in rows),
               "batch_size": len(y), "sequence_length": x.shape[1],
               "stage": "initial parameters, analytical gradients before clipping"}
    return summary, rows


def baselines(split):
    raw, y, pair_ids = split["x_raw"], split["y"], split["pair_ids"]
    # Сортировка гарантирует одинаковый порядок суммирования для пары:
    # исключаем ложные различия среднего из-за округления float64.
    means = np.mean(np.sort(raw[:, :, 0], axis=1), axis=1, keepdims=True)
    mean_predictions = means >= 0.0
    trend_predictions = raw[:, -1, :] > raw[:, 0, :]
    assert float(np.mean(mean_predictions == y)) == 0.5
    assert float(np.mean(trend_predictions == y)) == 1.0
    pair_id = int(pair_ids[0])
    indices = np.flatnonzero(pair_ids == pair_id)
    example = [{"label": int(y[i, 0]), "values": raw[i, :, 0].tolist(),
                "mean": float(means[i, 0]),
                "mean_prediction": int(mean_predictions[i, 0])} for i in indices]
    return {"chance_expected_accuracy": 0.5,
            "constant_class_accuracy": float(np.mean(y == 0)),
            "mean_threshold": 0.0,
            "mean_threshold_accuracy": float(np.mean(mean_predictions == y)),
            "first_last_trend_accuracy": float(np.mean(trend_predictions == y)),
            "paired_mean_proof": "Each pair has the same multiset and mean, opposite labels. Any deterministic classifier using only the mean predicts the same label for both, hence exactly one correct answer per pair (50%).",
            "example_pair_id": pair_id, "example_pair": example}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = {"dataset_seed": 24, "model_seed": 25, "shuffle_seed": 26,
                "train_pairs": 256, "test_pairs": 128, "length": 12,
                "hidden_units": 8, "epochs": 150, "batch_size": 64,
                "learning_rate": 0.01, "clip_global_norm": 1.0,
                "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1e-8}
    splits, scaling = make_pairs(settings["dataset_seed"], settings["train_pairs"], settings["test_pairs"], settings["length"])
    p = init_params(settings["model_seed"], settings["hidden_units"])
    gradient_check, gradient_rows = check_gradients(p, splits["train"]["x"][:4], splits["train"]["y"][:4])
    if not gradient_check["passed"]:
        raise RuntimeError(f"Gradient check failed: {gradient_check}")
    train = splits["train"]
    rng = np.random.default_rng(settings["shuffle_seed"])
    m, v = ({k: np.zeros_like(a) for k, a in p.items()} for _ in range(2))
    history = []
    initial_metrics, _ = evaluate(p, train)
    history.append({"epoch": 0, "train_loss": initial_metrics["loss"], "train_accuracy": initial_metrics["accuracy"]})
    step, clipped_steps, maximum_gradient_norm = 0, 0, 0.0
    for epoch in range(1, settings["epochs"] + 1):
        indices = rng.permutation(len(train["y"]))
        for start in range(0, len(indices), settings["batch_size"]):
            batch = indices[start:start + settings["batch_size"]]
            _, grads = loss_and_grad(p, train["x"][batch], train["y"][batch])
            norm = float(np.sqrt(sum(np.sum(g * g) for g in grads.values())))
            maximum_gradient_norm = max(maximum_gradient_norm, norm)
            scale = min(1.0, settings["clip_global_norm"] / (norm + 1e-12))
            clipped_steps += int(scale < 1.0)
            step += 1
            for key in p:
                g = grads[key] * scale
                m[key] = settings["adam_beta1"] * m[key] + (1.0 - settings["adam_beta1"]) * g
                v[key] = settings["adam_beta2"] * v[key] + (1.0 - settings["adam_beta2"]) * g * g
                mhat = m[key] / (1.0 - settings["adam_beta1"] ** step)
                vhat = v[key] / (1.0 - settings["adam_beta2"] ** step)
                p[key] -= settings["learning_rate"] * mhat / (np.sqrt(vhat) + settings["adam_epsilon"])
        metrics, _ = evaluate(p, train)
        history.append({"epoch": epoch, "train_loss": metrics["loss"], "train_accuracy": metrics["accuracy"]})
    final_metrics = {}
    prediction_rows, dataset_rows = [], []
    for split_name, split in splits.items():
        metrics, probs = evaluate(p, split)
        final_metrics[split_name] = metrics
        for i in range(len(split["y"])):
            common = {"split": split_name, "pair_id": int(split["pair_ids"][i]), "label": int(split["y"][i, 0])}
            prediction_rows.append({**common, "probability_ascending": float(probs[i, 0]), "prediction": int(probs[i, 0] >= 0.5)})
            dataset_rows.append({**common, **{f"x{t + 1}": float(split["x_raw"][i, t, 0]) for t in range(settings["length"])}})
    output = {"experiment": "Elman RNN, ascending versus reversed synthetic sequences",
              "numpy_version": np.__version__, "settings": settings, "scaling": scaling,
              "parameter_count": sum(a.size for a in p.values()), "initial_train": initial_metrics,
              "final": final_metrics, "baselines": {k: baselines(s) for k, s in splits.items()},
              "gradient_check": gradient_check,
              "optimization": {"updates": step, "clipped_updates": clipped_steps,
                               "max_global_gradient_norm_before_clipping": maximum_gradient_norm},
              "data_split": {"unit": "original forward/reverse pair", "train_examples": len(train["y"]),
                             "test_examples": len(splits["test"]["y"]), "overlap_pair_ids": [],
                             "test_used_for_training_or_selection": False},
              "limitations": ["Toy deterministic synthetic task; no claim about real-world quality.",
                              "A simple first/last rule reaches 100%; a neural network is unnecessary for this task.",
                              "One fixed seed, no cross-validation or confidence estimate.",
                              "Accuracy 100% on this test does not establish generalization to arbitrary sequences."]}
    (args.output_dir / "metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "training_history.csv", history)
    write_csv(args.output_dir / "gradient_check.csv", gradient_rows)
    write_csv(args.output_dir / "predictions.csv", prediction_rows)
    write_csv(args.output_dir / "dataset.csv", dataset_rows)
    np.savez(args.output_dir / "model_weights.npz", **p)
    print(json.dumps({"final": final_metrics, "gradient_check": gradient_check,
                      "optimization": output["optimization"], "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
