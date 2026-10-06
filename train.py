"""Select polynomial regressions, validate on a holdout, and predict both test sets."""
from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SEED = 42


def load_data(data_dir: Path, roll: str, variant: int):
    train = pd.read_csv(data_dir / f"{roll}_train_var{variant}.csv")
    test = pd.read_csv(data_dir / f"{roll}_test_var{variant}.csv")
    expected = [f"x{i}" for i in range(1, 7 if variant == 1 else 4)]
    if list(train.columns) != expected + ["y"] or list(test.columns) != expected:
        raise ValueError(f"Unexpected dataset schema for var{variant}")
    for frame in (train, test):
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("Data must contain finite numeric values")
    return train, test


def model():
    return Pipeline([
        ("poly", PolynomialFeatures(include_bias=False)),
        ("scale", StandardScaler()),
        ("reg", LinearRegression()),
    ])


def select_and_predict(train, test, variant, output, roll):
    y = train.y.to_numpy()
    fit_idx, hold_idx = train_test_split(np.arange(len(train)), test_size=0.2, random_state=SEED)
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    feature_sets = [["x1", "x2", "x3"], list(test.columns)] if variant == 1 else [["x1"], list(test.columns)]
    degrees = [range(1, 11), range(1, 7)] if variant == 1 else [range(1, 21), range(1, 21)]
    rows = []
    for features, degree_range in zip(feature_sets, degrees):
        X = train[features].to_numpy()
        grid = [
            {"poly__degree": list(degree_range), "reg": [LinearRegression()]},
            {"poly__degree": list(degree_range), "reg": [Ridge(solver="svd")],
             "reg__alpha": [0.0001, 0.01, 0.1, 1.0, 10.0, 100.0]},
        ]
        if features == list(test.columns):
            grid.append({"poly__degree": list(range(2, 7) if variant == 1 else range(4, 13)),
                         "reg": [Lasso(max_iter=50000, tol=1e-5)],
                         "reg__alpha": [0.001, 0.003, 0.01, 0.03, 0.1]})
            if variant == 1:
                grid.append({"poly__degree": [7, 8, 9, 10],
                             "reg": [Ridge(solver="lsqr", tol=1e-8, max_iter=10000)],
                             "reg__alpha": [1.0, 10.0, 100.0]})
        search = GridSearchCV(model(), grid, scoring="neg_mean_squared_error", cv=folds,
                              n_jobs=1, refit=False, error_score="raise")
        search.fit(X[fit_idx], y[fit_idx])
        for i, params in enumerate(search.cv_results_["params"]):
            degree = params["poly__degree"]
            rows.append({
                "variant": variant, "features": ",".join(features), "degree": degree,
                "terms": math.comb(len(features) + degree, degree) - 1,
                "alpha": float(params.get("reg__alpha", 0.0)),
                "regression": type(params["reg"]).__name__,
                "solver": getattr(params["reg"], "solver", "default"),
                "cv_mse": float(-search.cv_results_["mean_test_score"][i]),
                "cv_std": float(search.cv_results_["std_test_score"][i]),
                **{f"fold_{k+1}_mse": float(-search.cv_results_[f"split{k}_test_score"][i]) for k in range(5)},
            })
        print(f"var{variant}: evaluated {len(search.cv_results_['params'])} candidates using {features}", flush=True)
    results = pd.DataFrame(rows).sort_values("cv_mse")
    best = results.iloc[0]
    threshold = float(best.cv_mse + best.cv_std / np.sqrt(5))
    # Prefer a smaller polynomial if its score is within one standard error of the best.
    eligible = results[results.cv_mse <= threshold]
    chosen = eligible.sort_values(["terms", "cv_mse", "alpha"]).iloc[0]
    features = chosen.features.split(",")
    if chosen.regression == "LinearRegression":
        regressor = LinearRegression()
    elif chosen.regression == "Lasso":
        regressor = Lasso(alpha=float(chosen.alpha), max_iter=50000, tol=1e-5)
    else:
        regressor = Ridge(alpha=float(chosen.alpha), solver=chosen.solver, tol=1e-8, max_iter=10000)
    params = {"poly__degree": int(chosen.degree), "reg": regressor}
    selected = model().set_params(**params)
    X = train[features].to_numpy()
    selected.fit(X[fit_idx], y[fit_idx])
    pred_hold = selected.predict(X[hold_idx])
    hold_mse = mean_squared_error(y[hold_idx], pred_hold)
    hold_r2 = r2_score(y[hold_idx], pred_hold)
    pd.DataFrame({"training_row": hold_idx, "actual_y": y[hold_idx], "predicted_y": pred_hold,
                  "residual": y[hold_idx] - pred_hold}).to_csv(output / f"var{variant}_holdout.csv", index=False)
    baseline = np.full(len(hold_idx), y[fit_idx].mean())
    final_model = clone(selected).fit(X, y)
    predictions = final_model.predict(test[features].to_numpy())
    if len(predictions) != len(test) or not np.isfinite(predictions).all():
        raise ValueError("Invalid test predictions")
    prediction_path = ROOT / f"{roll}_pred_var{variant}.csv"
    pd.DataFrame({"y": predictions}).to_csv(prediction_path, index=False, float_format="%.15g")
    joblib.dump({"model": final_model, "features": features, "variant": variant}, output / f"var{variant}_model.joblib")
    poly = final_model.named_steps["poly"]
    scaler = final_model.named_steps["scale"]
    reg = final_model.named_steps["reg"]
    coefficients = reg.coef_ / scaler.scale_
    intercept = float(reg.intercept_ - np.dot(coefficients, scaler.mean_))
    pd.DataFrame({"term": ["1"] + list(poly.get_feature_names_out(features)),
                  "coefficient": [intercept] + coefficients.tolist()}).to_csv(output / f"var{variant}_coefficients.csv", index=False)
    results.to_csv(output / f"var{variant}_cv_results.csv", index=False)
    plot_results(results, chosen, y[hold_idx], pred_hold, variant, output)
    stats = {
        "variant": variant, "train_rows": len(train), "test_rows": len(test),
        "selection_rows": len(fit_idx), "holdout_rows": len(hold_idx),
        "features": features, "degree": int(chosen.degree), "terms": int(chosen.terms),
        "alpha": float(chosen.alpha), "candidates": len(results),
        "regression": chosen.regression, "nonzero_terms": int(np.count_nonzero(coefficients)),
        "cv_mse": float(chosen.cv_mse), "cv_std": float(chosen.cv_std),
        "best_cv_mse": float(best.cv_mse), "one_se_threshold": threshold,
        "holdout_mse": float(hold_mse), "holdout_rmse": float(np.sqrt(hold_mse)),
        "holdout_r2": float(hold_r2), "baseline_mse": float(mean_squared_error(y[hold_idx], baseline)),
        "full_train_mse": float(mean_squared_error(y, final_model.predict(X))),
        "prediction_min": float(predictions.min()), "prediction_max": float(predictions.max()),
        "intercept": intercept, "coefficients": dict(zip(poly.get_feature_names_out(features), coefficients.tolist())),
    }
    print(json.dumps({k: stats[k] for k in ["variant", "features", "degree", "alpha", "holdout_mse", "holdout_r2"]}), flush=True)
    return stats


def plot_results(results, chosen, actual, predicted, variant, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    line_styles = [("black", "o", "-"), ("#666666", "s", "--")]
    for i, (features, group) in enumerate(results.groupby("features", sort=False)):
        curve = group.groupby("degree").cv_mse.min()
        color, marker, linestyle = line_styles[i % len(line_styles)]
        axes[0].plot(curve.index, curve.values, color=color, marker=marker,
                     linestyle=linestyle, ms=4, label=features)
    axes[0].scatter([chosen.degree], [chosen.cv_mse], s=90, facecolors="none", edgecolors="black", zorder=5)
    axes[0].set(xlabel="Polynomial degree", ylabel="5-fold CV MSE", title="Degree and feature comparison", yscale="log")
    axes[0].legend(fontsize=8)
    axes[1].scatter(actual, predicted, s=12, alpha=.55, color="#333333")
    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    axes[1].plot(limits, limits, "--", color="black", lw=1)
    axes[1].set(xlabel="Actual y", ylabel="Predicted y", title="Untouched 20% holdout")
    fig.savefig(output / f"var{variant}_validation.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roll", default="BT2024186")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    data_dir = args.data_dir or ROOT / args.roll
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    if list(sample.columns) != ["y"]:
        raise ValueError("Sample submission must have only the y column")
    summary = {"seed": SEED, "python": platform.python_version(), "numpy": np.__version__,
               "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "problems": []}
    with threadpool_limits(limits=1):
        for variant in (1, 2):
            train, test = load_data(data_dir, args.roll, variant)
            if len(test) != len(sample):
                raise ValueError("Test and sample row counts differ")
            summary["problems"].append(select_and_predict(train, test, variant, args.output_dir, args.roll))
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
