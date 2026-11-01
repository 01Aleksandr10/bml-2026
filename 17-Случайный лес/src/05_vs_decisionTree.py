from _output import start_capture

_out = start_capture()

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Одно глубокое дерево (переобучается)
tree = DecisionTreeClassifier(max_depth=None, random_state=42).fit(X_train, y_train)
tree_acc = accuracy_score(y_test, tree.predict(X_test))

# Наращиваем число деревьев в RF
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

rf_scores = []
for n in range(1, 201, 5):
    proba = np.mean([t.predict_proba(X_test) for t in rf.estimators_[:n]], axis=0)
    rf_scores.append(accuracy_score(y_test, proba.argmax(axis=1)))

plt.figure(figsize=(8, 4))
plt.plot(range(1, 201, 5), rf_scores, label='Случайный лес', color='blue', linewidth=2)
plt.axhline(tree_acc, color='red', linestyle='--', label=f'Одно дерево (CART): {tree_acc:.3f}')
plt.xlabel('Число деревьев'); plt.ylabel('Accuracy'); plt.title('Насыщение качества RF')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()

here = os.path.dirname(os.path.abspath(__file__))
fig_path = os.path.join(here, "05_output.png")
plt.savefig(fig_path, dpi=120)
plt.close()

ns = list(range(1, 201, 5))
print(f"Accuracy одного дерева (CART):      {tree_acc:.4f}")
print(f"Accuracy RF ({ns[19]} деревьев):          {rf_scores[19]:.4f}")
print(f"Accuracy RF ({ns[-1]} деревьев):          {rf_scores[-1]:.4f}")
print(f"График сохранен: 05_output.png")

_out.save()
