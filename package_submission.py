"""Validate predictions and package the two prediction files and report."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from verify_outputs import main as verify


ROOT = Path(__file__).resolve().parent


def main():
    verify()
    paths = [ROOT / "BT2024186_pred_var1.csv", ROOT / "BT2024186_pred_var2.csv",
             ROOT / "output" / "pdf" / "BT2024186_report.pdf"]
    if any(not path.is_file() or path.stat().st_size == 0 for path in paths):
        raise ValueError("A submission deliverable is missing")
    folder = ROOT / "submission"
    folder.mkdir(exist_ok=True)
    destination = folder / "BT2024186_submission.zip"
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, arcname=path.name)
    with ZipFile(destination) as archive:
        assert archive.testzip() is None
    print(destination)


if __name__ == "__main__":
    main()
