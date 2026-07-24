import numpy as np
import pandas as pd

CHOP_GRID_S = 17
CHOP_GRID_T = 17
MAX_ROWS_PER_CELL = 1

def grid_chop_dataframe(
    df,
    n_s=CHOP_GRID_S,
    n_T=CHOP_GRID_T,
    max_rows_per_cell=MAX_ROWS_PER_CELL,
    seed=11,
):
    s_edges = np.linspace(float(df["s"].min()), float(df["s"].max()), n_s + 1)
    T_edges = np.linspace(float(df["T"].min()), float(df["T"].max()), n_T + 1)

    chopped_parts = []
    for s_idx in range(n_s):
        s_left = s_edges[s_idx]
        s_right = s_edges[s_idx + 1]
        s_mask = (df["s"] >= s_left) & (
            (df["s"] < s_right) if s_idx < n_s - 1 else (df["s"] <= s_right)
        )

        for T_idx in range(n_T):
            T_left = T_edges[T_idx]
            T_right = T_edges[T_idx + 1]
            T_mask = (df["T"] >= T_left) & (
                (df["T"] < T_right) if T_idx < n_T - 1 else (df["T"] <= T_right)
            )
            cell = df[s_mask & T_mask]
            if cell.empty:
                continue
            chopped_parts.append(
                cell.sample(
                    n=min(max_rows_per_cell, len(cell)),
                    random_state=seed + s_idx * n_T + T_idx,
                )
            )

    if not chopped_parts:
        raise ValueError("grid chop did not keep any data")

    return pd.concat(chopped_parts, ignore_index=True)


def structured_table_from_chopped_data(df, n_s=CHOP_GRID_S, n_T=CHOP_GRID_T):
    s_grid = np.linspace(float(df["s"].min()), float(df["s"].max()), n_s)
    T_grid = np.linspace(float(df["T"].min()), float(df["T"].max()), n_T)
    s_edges = np.linspace(float(df["s"].min()), float(df["s"].max()), n_s + 1)
    T_edges = np.linspace(float(df["T"].min()), float(df["T"].max()), n_T + 1)
    q_grid = np.full((n_s, n_T), np.nan, dtype=float)
    sigma_grid = np.full((n_s, n_T), np.nan, dtype=float)
    observation_mask = np.zeros((n_s, n_T), dtype=bool)

    has_sigma = "sigma" in df.columns
    
    for s_idx in range(n_s):
        s_left = s_edges[s_idx]
        s_right = s_edges[s_idx + 1]
        s_mask = (df["s"] >= s_left) & (
            (df["s"] < s_right) if s_idx < n_s - 1 else (df["s"] <= s_right)
        )

        for T_idx in range(n_T):
            T_left = T_edges[T_idx]
            T_right = T_edges[T_idx + 1]
            T_mask = (df["T"] >= T_left) & (
                (df["T"] < T_right) if T_idx < n_T - 1 else (df["T"] <= T_right)
            )
            cell = df[s_mask & T_mask]

            if cell.empty:
                continue
            
            observation_mask[s_idx, T_idx] = True
            q_grid[s_idx, T_idx] = float(cell["q_noisy"]).mean()
            if has_sigma:
                sigma_grid[s_idx, T_idx] = float(cell["sigma"].mean())
    
    observed_indices = np.argwhere(observation_mask)

    if observed_indices.size == 0:
        raise ValueError("structured grid contains no observed cells")

    missing_indices = np.argwhere(~observation_mask)

    for s_idx, T_idx in missing_indices:
        distances = (observed_indices[:, 0] - s_idx) ** 2 + (observed_indices[:, 1] - T_idx) ** 2
        nearest_index = int(np.argmin(distances))
        nearest_s, nearest_T = observed_indices[nearest_index]
        q_grid[s_idx, T_idx] = q_grid[nearest_s, nearest_T]
    
    if has_sigma:
        sigma_grid[~observation_mask] = 0.0
    else:
        sigma_grid = None

    return s_grid, T_grid, q_grid, observation_mask, sigma_grid

def scaled_structured_table_from_physical(s_grid, T_grid, q_grid, s_mean, s_std, T_mean, T_std):
    s_hat_grid = (s_grid - s_mean) / s_std
    T_hat_grid = (T_grid - T_mean) / T_std
    return s_hat_grid, T_hat_grid, q_grid