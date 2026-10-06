"""Check submission shape, row order, saved inference, and exported coefficients."""
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, train_test_split
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent


def main(check_convergence=False):
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    metrics = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
    for v in (1, 2):
        test = pd.read_csv(ROOT / "BT2024186" / f"BT2024186_test_var{v}.csv")
        submitted = pd.read_csv(ROOT / f"BT2024186_pred_var{v}.csv")
        assert list(submitted.columns) == list(sample.columns) == ["y"]
        assert len(submitted) == len(test) == len(sample) == 1000
        assert np.isfinite(submitted.to_numpy()).all()
        bundle = joblib.load(ROOT / "artifacts" / f"var{v}_model.joblib")
        model = bundle["model"]
        X = test[bundle["features"]].to_numpy()
        np.testing.assert_allclose(submitted.y, model.predict(X), rtol=1e-12, atol=1e-12)
        coeff = pd.read_csv(ROOT / "artifacts" / f"var{v}_coefficients.csv")
        raw_prediction = coeff.coefficient.iloc[0] + model.named_steps["poly"].transform(X) @ coeff.coefficient.iloc[1:].to_numpy()
        np.testing.assert_allclose(submitted.y, raw_prediction, rtol=1e-8, atol=1e-8)
        hold = pd.read_csv(ROOT / "artifacts" / f"var{v}_holdout.csv")
        assert len(hold) == 200 and hold.training_row.nunique() == 200
        summary = metrics["problems"][v-1]
        np.testing.assert_allclose(np.mean((hold.actual_y - hold.predicted_y) ** 2), summary["holdout_mse"], rtol=1e-12)
        scores = pd.read_csv(ROOT / "artifacts" / f"var{v}_cv_results.csv")
        assert len(scores) == summary["candidates"]
        assert np.isfinite(scores[["cv_mse", "cv_std"]].to_numpy()).all()
        # Weakly regularized search candidates can reach their iteration limit.
        # Verify every fold of the selected estimator actually converges.
        if check_convergence:
            train = pd.read_csv(ROOT / "BT2024186" / f"BT2024186_train_var{v}.csv")
            fit_idx, _ = train_test_split(np.arange(len(train)), test_size=0.2, random_state=42)
            features = train[bundle["features"]].to_numpy()[fit_idx]
            target = train.y.to_numpy()[fit_idx]
            fold_errors = []
            with warnings.catch_warnings(), threadpool_limits(limits=1):
                warnings.simplefilter("error", ConvergenceWarning)
                for fit, check in KFold(5, shuffle=True, random_state=42).split(features):
                    estimator = clone(model).fit(features[fit], target[fit])
                    fold_errors.append(np.mean((target[check] - estimator.predict(features[check])) ** 2))
            np.testing.assert_allclose(np.mean(fold_errors), summary["cv_mse"], rtol=1e-8, atol=1e-8)
        print(f"var{v}: verified 1,000 finite predictions, saved inference, coefficients and evaluation artifacts")


if __name__ == "__main__":
    main(check_convergence=True)
