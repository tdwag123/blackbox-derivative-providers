import numpy as np
import pandas as pd


def make_tabular_oracle(path, noisy=True):
    # NOTE: this could be more robust
    df = pd.read_csv(path)
    q_column = "q_noisy" if noisy else "q_true"
    required_columns = {"s", "T", q_column}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"tabular oracle CSV is missing required columns: {missing_text}")

    def oracle(s, T, return_full=False):
        s = np.asarray(s, dtype=float)
        T = np.asarray(T, dtype=float)

        distances = np.sqrt((df["s"] - s) ** 2 + (df["T"] - T) ** 2)
        closest = df.loc[distances.idxmin()]
        return float(closest[q_column])

    return oracle
