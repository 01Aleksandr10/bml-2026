#!/usr/bin/env python3
"""Три ручных шага скалярной RNN; использует только стандартную библиотеку."""
import json
import math
from pathlib import Path


def calculate(inputs=None):
    """Вычислить шаги для заданного входа; по умолчанию — [1, 0, -1]."""
    values = [1.0, 0.0, -1.0] if inputs is None else [float(x) for x in inputs]
    h = 0.0
    rows = []
    for t, x in enumerate(values, start=1):
        h_prev = h
        input_term = 0.5 * x
        memory_term = 0.8 * h_prev
        preactivation = input_term + memory_term
        h = math.tanh(preactivation)
        rows.append({"t": t, "x": x, "h_previous": h_prev,
                     "input_term_0.5x": input_term, "memory_term_0.8h": memory_term,
                     "preactivation": preactivation, "h": h})
    return {"formula": "h_t = tanh(0.5*x_t + 0.8*h_(t-1))", "h_0": 0.0,
            "inputs": values, "steps": rows, "final_h": h}


if __name__ == "__main__":
    result = calculate()
    result["reversed_example"] = calculate(reversed(result["inputs"]))
    output = Path(__file__).resolve().parents[1] / "results" / "numerical_example.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
