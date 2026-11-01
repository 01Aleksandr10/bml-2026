from _output import start_capture

_out = start_capture()

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, random_state=42)

# Чистый Bagging (берет все признаки)
bag = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50,
                        max_features=1.0, random_state=42).fit(X, y)
# Случайный лес (берет sqrt(20) ≈ 4 признака)
rf = RandomForestClassifier(n_estimators=50, max_features='sqrt', random_state=42).fit(X, y)

# Считаем попарную корреляцию предсказаний деревьев
def get_corr(model):
    preds = np.array([t.predict(X) for t in model.estimators_])
    corr_matrix = np.corrcoef(preds)
    # Берем среднее по верхней треугольной матрице (без диагонали)
    return corr_matrix[np.triu_indices_from(corr_matrix, k=1)].mean()

print(f"Средняя корреляция деревьев в Bagging (max_features=1.0): {get_corr(bag):.3f}")
print(f"Средняя корреляция деревьев в RF (max_features='sqrt'):  {get_corr(rf):.3f}")
# Вывод: корреляция в RF ниже, чем в Bagging, что снижает дисперсию ансамбля.

_out.save()
