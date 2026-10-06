"""Run inference with a saved polynomial model without retraining."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    data = pd.read_csv(args.input)
    values = data[bundle["features"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Input contains nonfinite values")
    with threadpool_limits(limits=1):
        pred = bundle["model"].predict(values)
    if not np.isfinite(pred).all():
        raise ValueError("Predictions contain nonfinite values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y": pred}).to_csv(args.output, index=False, float_format="%.15g")


if __name__ == "__main__":
    main()
