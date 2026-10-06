"""Build the four-page assignment report from measured training artifacts."""
from __future__ import annotations
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "pdf"
INK = colors.black
ACCENT = colors.black
LIGHT = colors.HexColor("#F2F2F2")
RULE = colors.HexColor("#BFBFBF")


def main():
    metrics = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
    font_dir = Path("C:/Windows/Fonts")
    if (font_dir / "arial.ttf").exists():
        pdfmetrics.registerFont(TTFont("Report", str(font_dir / "arial.ttf")))
        pdfmetrics.registerFont(TTFont("ReportBold", str(font_dir / "arialbd.ttf")))
        pdfmetrics.registerFontFamily("Report", normal="Report", bold="ReportBold")
    else:
        pdfmetrics.registerFontFamily("Report", normal="Helvetica", bold="Helvetica-Bold")
        # Portable fallback uses the PDF's built-in fonts.
    base_font = "Report" if "Report" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
    bold_font = "ReportBold" if "ReportBold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("BodyCustom", fontName=base_font, fontSize=10.1, leading=14.3,
                              textColor=INK, spaceAfter=8))
    styles.add(ParagraphStyle("TitleCustom", fontName=bold_font, fontSize=27, leading=31,
                              textColor=INK, spaceAfter=12))
    styles.add(ParagraphStyle("HeadingCustom", fontName=bold_font, fontSize=14, leading=18,
                              textColor=ACCENT, spaceBefore=10, spaceAfter=7))
    styles.add(ParagraphStyle("SmallCustom", fontName=base_font, fontSize=8.6, leading=11.5,
                              textColor=INK, spaceAfter=6))
    styles.add(ParagraphStyle("CellCustom", fontName=base_font, fontSize=8.5, leading=11,
                              textColor=INK))
    story = []

    def p(text, style="BodyCustom"):
        return Paragraph(text, styles[style])

    def add(text, style="BodyCustom"):
        story.append(p(text, style))

    def table(rows, widths):
        cells = [[p(escape(str(value)), "CellCustom") for value in row] for row in rows]
        item = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LINEBELOW", (0, 0), (-1, 0), .7, ACCENT),
            ("LINEBELOW", (0, 1), (-1, -1), .3, RULE),
        ]))
        story.append(item)
        story.append(Spacer(1, 8))

    add("POLYNOMIAL REGRESSION", "SmallCustom")
    add("Geothermal power<br/>and reservoir prediction", "TitleCustom")
    add("Machine Learning Assignment 1 | Roll number: <b>BT2024186</b>")
    add("1. Data and approach", "HeadingCustom")
    add("The two personalised problems were modelled separately. Each has 1,000 labelled training rows "
        "and 1,000 unlabelled test rows. Var1 supplies six operational inputs; var2 supplies three spatial "
        "inputs. All values are finite, no values are missing, and inputs are bounded between -1 and 1. "
        "The original preprocessing was retained.")
    add("Each model uses all monomials whose total degree is at most d, including feature interactions. "
        "The fitted function is <b>y_hat = b + sum(beta_j * phi_j(x))</b>, where phi_j is a polynomial "
        "term. PolynomialFeatures excludes a constant column because the regression fits an intercept. "
        "Expanded columns are standardized within the training pipeline to improve numerical stability.")
    add("2. Selection and independent evaluation", "HeadingCustom")
    add("A reproducible 80/20 split (seed 42) reserves 200 rows per problem for holdout evaluation. "
        "On the remaining 800 rows, shuffled five-fold cross-validation compares degrees, feature sets, "
        "and ordinary least squares, ridge, and lasso regression. The scaler is fitted inside each fold. "
        "Candidate selection uses only cross-validation MSE; holdout scores and test inputs do not select "
        "hyperparameters. Final models are refitted on all 1,000 labelled rows.")
    add("The one-standard-error rule selects the polynomial with the fewest expanded terms among "
        "candidates with MSE no greater than the minimum CV MSE plus the best candidate's fold standard "
        "deviation divided by sqrt(5). Within that term count, the lowest MSE wins. This is a practical "
        "complexity heuristic, rather than a formal confidence interval.")
    rows = [["Problem", "Degree / inputs", "Holdout MSE", "Holdout R2"]]
    for s in metrics["problems"]:
        rows.append([f"var{s['variant']}", f"{s['degree']} / {len(s['features'])} inputs",
                     f"{s['holdout_mse']:.6f}", f"{s['holdout_r2']:.6f}"])
    table(rows, [65, 160, 135, 130])
    add("These are validation results on training data, not scores against the hidden test ground truth.", "SmallCustom")

    for s in metrics["problems"]:
        v = s["variant"]
        story.append(PageBreak())
        add(f"PROBLEM {v}", "SmallCustom")
        add("Steam turbine optimization" if v == 1 else "Thermal reservoir mapping", "TitleCustom")
        add(f"Selected: <b>degree {s['degree']}</b>, inputs <b>{', '.join(s['features'])}</b>, "
            f"<b>{s['regression']}</b> with alpha = <b>{s['alpha']:g}</b>.")
        add("3. Search and rationale" if v == 1 else "4. Search and rationale", "HeadingCustom")
        if v == 1:
            add("The search compared x1-x3 at degrees 1-10 with all six inputs at degrees 1-10. "
                "OLS and SVD ridge were evaluated through degree 6 for all inputs; higher degrees used "
                "iterative ridge with alpha 1, 10, or 100 to handle the large design matrices. Lasso was "
                "evaluated at degrees 2-6. The reduced feature set was tested explicitly.")
        else:
            add("The search compared x1 alone with all three inputs at every degree from 1 to 20. "
                "OLS and SVD ridge were evaluated for both feature sets; lasso was evaluated on all "
                "three inputs at degrees 4-12. The single-feature alternative was tested explicitly.")
        add(f"A total of {s['candidates']} candidates were evaluated. The chosen model has "
            f"{s['terms']} expanded terms excluding the intercept, of which {s['nonzero_terms']} have "
            f"nonzero fitted coefficients. Its CV MSE is {s['cv_mse']:.6f} with fold standard deviation "
            f"{s['cv_std']:.6f}; the minimum CV MSE is {s['best_cv_mse']:.6f}. The one-standard-error "
            f"threshold is {s['one_se_threshold']:.6f}.")
        cv = pd.read_csv(ROOT / "artifacts" / f"var{v}_cv_results.csv")
        selected_features = ",".join(s["features"])
        degrees = sorted(set([1, max(1, s["degree"] - 1), s["degree"], min(10 if v == 1 else 20, s["degree"] + 1)]))
        rows = [["Inputs / degree", "Best regression", "Alpha", "CV MSE"]]
        for d in degrees:
            row = cv[(cv.features == selected_features) & (cv.degree == d)].sort_values("cv_mse").iloc[0]
            rows.append([f"All inputs / {d}", row.regression, f"{row.alpha:g}", f"{row.cv_mse:.6f}"])
        reduced = cv[cv.features != selected_features].sort_values("cv_mse")
        if len(reduced):
            row = reduced.iloc[0]
            rows.append([f"Reduced inputs / {row.degree}", row.regression, f"{row.alpha:g}", f"{row.cv_mse:.6f}"])
        table(rows, [155, 140, 65, 130])
        story.append(Image(str(ROOT / "artifacts" / f"var{v}_validation.png"), width=490, height=181.3))
        add("Left: minimum CV MSE across tested regressors at each degree; circle marks the selected "
            "model. Right: holdout predictions against observations, with a perfect-prediction reference line.", "SmallCustom")
        add(f"Holdout MSE = <b>{s['holdout_mse']:.6f}</b>; RMSE = <b>{s['holdout_rmse']:.6f}</b>; "
            f"R2 = <b>{s['holdout_r2']:.6f}</b>. A mean-only baseline has holdout MSE "
            f"{s['baseline_mse']:.6f}. The final full-data training MSE is {s['full_train_mse']:.6f}; "
            "it is a fitting diagnostic and does not replace validation.")

    story.append(PageBreak())
    add("REPRODUCIBILITY", "SmallCustom")
    add("Implementation and deliverables", "TitleCustom")
    add("5. Regularization and numerical choices", "HeadingCustom")
    add("Ridge minimizes squared residual error plus alpha times the squared coefficient norm. "
        "Lasso instead penalizes the absolute coefficient norm and can remove unnecessary monomials. "
        "Both remain polynomial regression: prediction is a linear combination of polynomial terms. "
        "The intercept is unpenalized. Ridge alphas are 0.0001, 0.01, 0.1, 1, 10, and 100; lasso "
        "alphas are 0.001, 0.003, 0.01, 0.03, and 0.1. Large var1 expansions use a narrower ridge grid.")
    add("SVD ridge handles correlated polynomial columns; high-degree var1 ridge uses LSQR with "
        "tolerance 1e-8 and at most 10,000 iterations. Lasso uses tolerance 1e-5 and at most 50,000 "
        "iterations. Some high-degree lasso candidates reached the iteration limit; both selected "
        "models were checked to converge on every CV fold. BLAS threads are limited to one for "
        "repeatability and to avoid oversubscription.")
    add("6. Code and reproduction", "HeadingCustom")
    add(f"Measured environment: Python {metrics['python']}; NumPy {metrics['numpy']}; pandas "
        f"{metrics['pandas']}; scikit-learn {metrics['scikit_learn']}. No GPU is required. Run "
        "<b>python train.py</b> from the repository root to repeat selection, validation, final fitting, "
        "and prediction. Run <b>python build_report.py</b> after training to regenerate this report.")
    add("For inference without retraining, run <b>python predict.py --model artifacts/var1_model.joblib "
        "--input BT2024186/BT2024186_test_var1.csv --output predictions.csv</b>. Use the analogous var2 "
        "paths for the second problem. README.md contains Windows commands and dependency instructions.")
    add('Repository: <link href="https://github.com/AadyantNeog/MLassignment1" color="#000000">'
        'github.com/AadyantNeog/MLassignment1</link>. All model selection and inference code is supplied.')
    table([["Deliverable", "Contents"],
           ["BT2024186_pred_var1.csv", "1,000 predictions; a single y column; original test row order"],
           ["BT2024186_pred_var2.csv", "1,000 predictions; a single y column; original test row order"],
           ["BT2024186_report.pdf", "This four-page report"],
           ["artifacts/", "Saved models, coefficients, fold scores, holdout predictions, metrics and plots"]], [180, 310])
    add("7. Validation and limitations", "HeadingCustom")
    add("The prediction files follow the provided sample submission: header y, no index and no added "
        "feature columns. Each row corresponds directly to the same row in its original test CSV. "
        "Saved-model predictions are checked against the submitted files, and polynomial coefficients "
        "are exported in the original input basis for inspection.")
    add("Hidden test targets are unavailable, so test MSE and R2 cannot be computed locally. The reported "
        "holdout uses one reproducible random split and has sampling uncertainty. Degree and feature "
        "choices are justified by cross-validation on these personalised datasets; no external data, "
        "non-polynomial predictor, test labels, or target-based preprocessing is used.")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "BT2024186_report.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=(595.28, 841.89), rightMargin=52.64, leftMargin=52.64,
                            topMargin=48, bottomMargin=46, title="Polynomial Regression - BT2024186",
                            author="BT2024186")

    def footer(canvas, doc):
        canvas.setStrokeColor(RULE)
        canvas.line(52.64, 35, 542.64, 35)
        canvas.setFont(base_font, 8)
        canvas.setFillColor(INK)
        canvas.drawString(52.64, 22, "BT2024186 | ML Assignment 1")
        canvas.drawRightString(542.64, 22, str(doc.page))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    if doc.page != 4:
        raise ValueError(f"Expected a four-page report, generated {doc.page} pages")
    print(path)


if __name__ == "__main__":
    main()
