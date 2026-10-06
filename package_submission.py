"""Validate predictions and package the three submission files with the repo URL."""
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
    repository = folder / "REPOSITORY.txt"
    repository.write_text("https://github.com/AadyantNeog/MLassignment1\n", encoding="utf-8")
    destination = folder / "BT2024186_submission.zip"
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for path in paths + [repository]:
            archive.write(path, arcname=path.name)
    with ZipFile(destination) as archive:
        assert archive.testzip() is None
    print(destination)


if __name__ == "__main__":
    main()
