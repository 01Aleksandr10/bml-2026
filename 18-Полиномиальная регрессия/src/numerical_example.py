"""МНК для пяти наблюдений: матрица признаков, коэффициенты и остатки."""
import json
from pathlib import Path
import numpy as np


class PolynomialRegressor:
    """Одномерный МНК. Столбец единиц входит в матрицу признаков."""

    def __init__(self, degree=2):
        self.degree = degree

    def fit(self, x, y):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(x) != len(y) or len(x) == 0:
            raise ValueError("x and y must have equal nonzero length")
        if not isinstance(self.degree, int) or self.degree < 0:
            raise ValueError("degree must be a nonnegative integer")
        phi = np.vander(x, N=self.degree + 1, increasing=True)
        self.coef_, _, self.rank_, self.singular_values_ = np.linalg.lstsq(
            phi, y, rcond=None
        )
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        phi = np.vander(x, N=self.degree + 1, increasing=True)
        return phi @ self.coef_


def run():
    x = np.array([-2., -1., 0., 1., 2.])
    y = np.array([5., 2., 1., 2., 6.])
    phi = np.vander(x, 3, increasing=True)
    quadratic = PolynomialRegressor(2).fit(x, y)
    linear = PolynomialRegressor(1).fit(x, y)
    pred = quadratic.predict(x)
    result = {
        "x": x.tolist(), "y": y.tolist(), "phi": phi.tolist(),
        "phi_t_phi": (phi.T @ phi).tolist(), "phi_t_y": (phi.T @ y).tolist(),
        "coef": quadratic.coef_.tolist(), "predictions": pred.tolist(),
        "residuals": (y - pred).tolist(),
        "sse": float(np.sum((y - pred)**2)),
        "rmse": float(np.sqrt(np.mean((y - pred)**2))),
        "linear_coef": linear.coef_.tolist(),
        "linear_sse": float(np.sum((y - linear.predict(x))**2)),
        "linear_rmse": float(np.sqrt(np.mean((y - linear.predict(x))**2))),
        "new_x": 1.5, "new_prediction": float(quadratic.predict([1.5])[0]),
    }
    np.testing.assert_allclose(quadratic.coef_, [32/35, 1/5, 8/7], atol=1e-12)
    np.testing.assert_allclose(phi.T @ (y-pred), 0, atol=1e-12)
    print("Phi:\n", phi)
    print("Phi.T @ Phi:\n", phi.T @ phi)
    print("Phi.T @ y:", phi.T @ y)
    print("Коэффициенты:", quadratic.coef_)
    print("Прогнозы:", pred)
    print(f"RMSE параболы: {result['rmse']:.4f}")
    print(f"RMSE прямой: {result['linear_rmse']:.4f}")
    print(f"Прогноз при x = 1.5: {result['new_prediction']:.4f}")
    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    (out / "numerical.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
