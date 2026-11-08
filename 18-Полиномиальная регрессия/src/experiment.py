"""Подбор степени полинома и сравнение моделей на одной выборке."""

import csv
import json
from pathlib import Path
import platform
import numpy as np
import scipy
import sklearn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, SplineTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from numerical_example import PolynomialRegressor


def make_models():
    poly = Pipeline([
        ("poly", PolynomialFeatures(include_bias=False)),
        ("scale", StandardScaler()),
        ("ridge", Ridge()),
    ])
    specifications = [
        ("Линейная", LinearRegression(), {}),
        ("Полином + Ridge", poly, {
            "poly__degree": [1, 2, 3, 5, 8, 12],
            "ridge__alpha": [.001, .01, .1, 1., 10.],
        }),
        ("Кубический сплайн", Pipeline([
            ("spline", SplineTransformer(degree=3, include_bias=False)),
            ("scale", StandardScaler()), ("ridge", Ridge()),
        ]), {"spline__n_knots": [3, 5, 7], "ridge__alpha": [.01, .1, 1., 10.]}),
        ("Дерево", DecisionTreeRegressor(random_state=18), {
            "max_depth": [2, 3, 5, None], "min_samples_leaf": [2, 5, 10]
        }),
        ("SVR с RBF", Pipeline([
            ("scale", StandardScaler()), ("svr", SVR())
        ]), {"svr__C": [1., 10., 100.], "svr__gamma": [.1, 1., 10.],
             "svr__epsilon": [.1, .3]}),
    ]
    return specifications


def save_results(payload, out):
    (out/"experiment.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out/"comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload["results"][0]))
        writer.writeheader()
        writer.writerows(payload["results"])
    with (out/"data.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "x", "y"])
        writer.writerows(["train", float(x), float(y)] for x,y in zip(payload["X_train"], payload["y_train"]))
        writer.writerows(["test", float(x), float(y)] for x,y in zip(payload["X_test"], payload["y_test"]))


def main():
    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(18)
    X = rng.uniform(-2, 2, size=(80, 1))
    y = 1 + .6 * X[:, 0] + 1.2 * X[:, 0]**2 + rng.normal(0, .6, 80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=18
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=18)
    specifications = make_models()
    results = []
    models = {}
    for name, model, grid in specifications:
        search = GridSearchCV(model, grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=1)
        search.fit(X_train, y_train)
        pred = search.predict(X_test)
        row = {
            "model": name, "cv_rmse": float(np.sqrt(-search.best_score_)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, search.predict(X_train)))),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "test_r2": float(r2_score(y_test, pred)), "params": search.best_params_,
        }
        results.append(row)
        models[name] = search.best_estimator_
        print(f"{name}: CV RMSE = {row['cv_rmse']:.3f}, "
              f"test RMSE = {row['test_rmse']:.3f}")
        if name == "Полином + Ridge":
            best_search = search
    # МНК без регуляризации для сравнения степеней на графике.
    x_grid = np.linspace(-2, 2, 81)
    curves = {"x": x_grid.tolist(), "truth": (1+.6*x_grid+1.2*x_grid**2).tolist()}
    degree_rows = []
    for degree in [1, 2, 12]:
        fit = PolynomialRegressor(degree).fit(X_train[:, 0], y_train)
        curves[f"degree_{degree}"] = fit.predict(x_grid).tolist()
        degree_rows.append({"degree": degree,
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, fit.predict(X_train[:, 0])))),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, fit.predict(X_test[:, 0]))))})
    curves["selected"] = models["Полином + Ridge"].predict(x_grid[:, None]).tolist()
    # Для каждой степени выбираем alpha с минимальной ошибкой CV.
    cv_rows = []
    cv_results = best_search.cv_results_
    for degree in [1, 2, 3, 5, 8, 12]:
        ids = [i for i,p in enumerate(cv_results["params"]) if p["poly__degree"] == degree]
        idx = max(ids, key=lambda i: cv_results["mean_test_score"][i])
        cv_rows.append({"degree": degree,
            "cv_rmse": float(np.sqrt(-cv_results["mean_test_score"][idx])),
            "alpha": cv_results["params"][idx]["ridge__alpha"]})
    for degree in [1, 2]:
        a = PolynomialRegressor(degree).fit(np.array([-2,-1,0,1,2]), np.array([5,2,1,2,6]))
        b = Pipeline([("poly", PolynomialFeatures(degree, include_bias=False)),
                      ("linear", LinearRegression())]).fit(
                          np.array([-2,-1,0,1,2])[:, None], np.array([5,2,1,2,6]))
        np.testing.assert_allclose(a.predict(x_grid), b.predict(x_grid[:, None]), atol=1e-10)
    payload = {"seed": 18, "n": 80, "n_train": len(X_train), "n_test": len(X_test),
        "noise_sigma": .6, "results": results, "curves": curves, "degrees": degree_rows,
        "cv_degrees": cv_rows,
        "X_train": X_train[:, 0].tolist(), "y_train": y_train.tolist(),
        "X_test": X_test[:, 0].tolist(), "y_test": y_test.tolist(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}}
    save_results(payload, out)



if __name__ == "__main__":
    main()
