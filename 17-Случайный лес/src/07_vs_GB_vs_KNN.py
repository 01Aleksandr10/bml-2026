from _output import start_capture

_out = start_capture()

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Добавляем 50 шумовых признаков и НЕ масштабируем данные (убиваем k-NN)
rng = np.random.RandomState(42)
X_train_dirty = np.hstack([X_train, rng.randn(len(X_train), 50) * 100])
X_test_dirty = np.hstack([X_test, rng.randn(len(X_test), 50) * 100])

models = {
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

results = []
for name, model in models.items():
    t0 = time.time()
    model.fit(X_train_dirty, y_train)
    train_time = time.time() - t0
    acc = accuracy_score(y_test, model.predict(X_test_dirty))
    results.append({"Метод": name, "Accuracy": f"{acc:.4f}", "Время (с)": f"{train_time:.3f}"})

df = pd.DataFrame(results)
print(df.to_markdown(index=False))
# Вывод: k-NN падает из-за шума и масштаба. GB точен, но медленнее. RF — золотая середина: быстро и робастно.

_out.save()
