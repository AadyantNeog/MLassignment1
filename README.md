# Polynomial Regression - ML Assignment 1

Roll number: **BT2024186**. This repository solves both personalised geothermal
regression problems using only polynomial regression.

The two prediction files contain one `y` column and 1,000 predictions each, in
the original test row order. The report is in
[`output/pdf/BT2024186_report.pdf`](output/pdf/BT2024186_report.pdf).

## Approach

- Keep the two problems separate and retain their supplied preprocessing.
- Reserve 20% of each training set for holdout evaluation (`random_state=42`).
- Use shuffled five-fold cross-validation on the other 80% to compare polynomial
  degrees, feature sets, ordinary least squares, ridge and lasso.
- Fit standardization inside each fold, after generating all monomials of total
  degree at most the candidate degree. The intercept is fitted separately.
- Prefer the smallest expansion whose CV MSE is within one standard error of the
  best CV score. Within that expansion size, choose the lowest CV MSE.
- Evaluate the selected model on the reserved holdout, then refit on all labelled
  rows and predict the provided test inputs.

Var1 compares the first three inputs with all six, through degree 10. The largest
six-input expansions use regularized iterative ridge; OLS/SVD ridge and lasso
use smaller expansions. Var2 compares `x1` alone with all three inputs through
degree 20, with additional lasso candidates through degree 12. Full candidate
grids, fold scores and chosen models are recorded under `artifacts/`.

The holdout is excluded from model selection. Test targets are hidden, so the
reported MSE and R2 values are validation estimates, not hidden-test scores.

| Problem | Selected model | Holdout MSE | Holdout R2 |
| --- | --- | --- | --- |
| var1 | Degree 5, all six inputs, lasso alpha 0.01 | 0.383402 | 0.964099 |
| var2 | Degree 10, all three inputs, lasso alpha 0.001 | 0.273390 | 0.994922 |

Some high-degree lasso candidates reached their iteration limit. The chosen
models were checked to converge on every cross-validation fold. Those fold
scores also reproduce the recorded selection scores.

## Run on Windows

Your existing canonical Python already has the ML dependencies. No environment
replacement or PyTorch installation is required:

```powershell
$mlPython = 'C:\Users\Aadyant Neog\AppData\Local\Programs\Python\Python313\python.exe'
& $mlPython train.py
& $mlPython verify_outputs.py
```

If setting up on another machine, install `requirements.txt` into a compatible
Python environment (the measured ML environment uses Python 3.13.1). On this
machine, optional isolation can reuse the existing packages:

```powershell
& $mlPython -m venv --system-site-packages .venv
```

Report generation also needs ReportLab. The Codex bundled document Python was
used to build the submitted report because ReportLab is available there:

```powershell
$documentPython = 'C:\Users\Aadyant Neog\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $documentPython build_report.py
```

On another machine with the declared packages installed, `python build_report.py`
is sufficient. The report reads the saved metrics and plots and performs no
training.

## Inference without retraining

```powershell
& $mlPython predict.py --model artifacts/var1_model.joblib --input BT2024186/BT2024186_test_var1.csv --output predictions_var1.csv
& $mlPython predict.py --model artifacts/var2_model.joblib --input BT2024186/BT2024186_test_var2.csv --output predictions_var2.csv
```

`train.py` also accepts `--roll`, `--data-dir`, and `--output-dir`. Defaults use
the supplied `BT2024186` inputs. The provided sample submission is checked for
its column name and row count.

## Files

| Path | Purpose |
| --- | --- |
| `train.py` | Model search, holdout evaluation, final fitting, predictions and plots |
| `predict.py` | Load a saved model and run inference |
| `verify_outputs.py` | Check submission schema, row order, inference and polynomial coefficients |
| `build_report.py` | Generate the four-page PDF from the measured results |
| `package_submission.py` | Verify and bundle the report, predictions and repository URL |
| `BT2024186/` | Original training and test datasets |
| `BT2024186_pred_var1.csv`, `BT2024186_pred_var2.csv` | Submission predictions |
| `artifacts/metrics.json` | Selected models, validation results and environment versions |
| `artifacts/var*_cv_results.csv` | Every candidate's MSE and individual fold scores |
| `artifacts/var*_holdout.csv` | Actual and predicted values on the reserved 200 rows |
| `artifacts/var*_coefficients.csv` | Polynomial coefficients in the original input basis |
| `artifacts/var*_model.joblib` | Full-data fitted models and their feature lists |
| `output/pdf/BT2024186_report.pdf` | Submission report |

`verify_outputs.py` checks that all submitted values are finite and match the
saved models on the original test rows. It also verifies the exported
coefficients and recorded holdout MSE.

After building the report, run `python package_submission.py` to recreate
`submission/BT2024186_submission.zip`. It contains both prediction CSVs, the
report and a text file with the repository URL.
