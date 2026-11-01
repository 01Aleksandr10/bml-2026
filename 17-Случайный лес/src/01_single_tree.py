from _output import start_capture

_out = start_capture()

import numpy as np
import pandas as pd

# Микро-выборка: 8 объектов, 2 признака, бинарный класс
data = pd.DataFrame({
    'X1': [1, 2, 3, 4, 5, 6, 7, 8],
    'X2': [2, 4, 1, 3, 5, 2, 4, 1],
    'y':  [0, 0, 0, 0, 1, 1, 1, 1]
})

np.random.seed(42)
# Шаг 1: Бутстрап (выборка с возвращением)
bootstrap_idx = np.random.choice(len(data), size=len(data), replace=True)
print("1. Индексы бутстрап-выборки:", bootstrap_idx)
# Выпали: [6 3 4 6 2 7 4 4] (объекты 4, 6 и 7 встретились дважды/трижды, 0, 1 и 5 — ни разу)

# Шаг 2: OOB объекты (те, что не попали в выборку)
oob_idx = np.setdiff1d(np.arange(len(data)), bootstrap_idx)
print("2. Индексы OOB (для валидации):", oob_idx)
# OOB: [0, 1, 5]

# Шаг 3: Формирование выборки для дерева
tree_data = data.iloc[bootstrap_idx]
print("\n3. Обучающая выборка для одного дерева:\n", tree_data[['X1', 'y']])

_out.save()
