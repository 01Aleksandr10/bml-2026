from _output import start_capture

_out = start_capture()

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import numpy as np

# Создаем данные: 1 полезный признак и 1 случайный, но с 50 уникальными значениями
rng = np.random.RandomState(42)
X_useful = rng.randn(1000, 1)
y = (X_useful[:, 0] > 0).astype(int)
X_noise = rng.randint(0, 50, size=(1000, 1)) # Шум с высокой кардинальностью

X = np.hstack([X_useful, X_noise])
rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)

gini_imp = rf.feature_importances_
perm_imp = permutation_importance(rf, X, y, n_repeats=10, random_state=42)

print("Важность по Gini (может завышать вклад шумовых признаков):")
print(f"  Полезный: {gini_imp[0]:.3f}, Шумовой: {gini_imp[1]:.3f}")
print("\nPermutation Importance (объективная):")
print(f"  Полезный: {perm_imp.importances_mean[0]:.3f}, Шумовой: {perm_imp.importances_mean[1]:.3f}")

_out.save()
