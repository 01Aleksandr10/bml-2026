from _output import start_capture

_out = start_capture()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

print(f"OOB-оценка (внутренняя, без отложенной выборки): {rf.oob_score_:.4f}")
print(f"Accuracy на hold-out (30% данных):               {accuracy_score(y_test, rf.predict(X_test)):.4f}")
# OOB использует ~63% уникальных объектов. Это "бесплатная" проверка.

_out.save()
