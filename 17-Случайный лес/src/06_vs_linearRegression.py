from _output import start_capture

_out = start_capture()

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

rng = np.random.RandomState(42)
X_train = np.linspace(1, 10, 50).reshape(-1, 1)
y_train = 2 * X_train.ravel() + rng.randn(50) * 0.5

X_test = np.linspace(11, 15, 20).reshape(-1, 1) # Зона экстраполяции
y_test_true = 2 * X_test.ravel()

rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)
lin = LinearRegression().fit(X_train, y_train)

plt.figure(figsize=(8, 4))
plt.scatter(X_train, y_train, label='Обучение', color='blue', alpha=0.6, zorder=5)
plt.plot(X_test, y_test_true, 'k--', label='Истина (y=2x)', linewidth=2)
plt.plot(X_test, lin.predict(X_test), 'g-', label='Линейная регрессия', linewidth=2)
plt.plot(X_test, rf.predict(X_test), 'r-', label='Случайный лес', linewidth=2)
plt.axvline(10, color='gray', linestyle=':', alpha=0.7)
plt.text(10.1, plt.ylim()[1]*0.95, '← обучение | экстраполяция →', color='gray')
plt.legend(); plt.title('RF не экстраполирует: прогноз "застревает"'); plt.grid(alpha=0.3)
plt.tight_layout()

here = os.path.dirname(os.path.abspath(__file__))
fig_path = os.path.join(here, "06_output.png")
plt.savefig(fig_path, dpi=120)
plt.close()

lin_mae = np.abs(lin.predict(X_test) - y_test_true).mean()
rf_mae = np.abs(rf.predict(X_test) - y_test_true).mean()
print(f"MAE линейной регрессии в зоне экстраполяции: {lin_mae:.4f}")
print(f"MAE случайного леса в зоне экстраполяции:   {rf_mae:.4f}")
print(f"График сохранен: 06_output.png")

_out.save()
