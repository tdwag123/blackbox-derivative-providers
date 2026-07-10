import sys
from pathlib import Path
import time

import numpy as np

from data import structured_table_from_chopped_data, scaled_structured_table_from_physical
from tabular_models import (
    LocalSavGolProvider,
    SAVGOL_NEIGHBORS,
    TabularFiniteDifferenceProvider,
    project_table_nonincreasing_in_s,
    smooth_table_3x3,
)

ROOT = Path(__file__).resolve().parents[1]
TABULAR_DIR = ROOT / "TabularDataMethods"
sys.path.append(str(ROOT))

from Methods.AnalyticReference import a_true, b_true, q_true
from Methods.TabularDataMethods.CubicSplines import CubicSplineFluxST
from Methods.TabularDataMethods.GaussianProcessesWrap import KISSGPFluxST
from Methods.TabularDataMethods.MonotoneInterpolation import PchipFluxST
from Methods.TabularDataMethods.RadialBasisFunctions import RBFDerivativeProviderST
from Methods.TabularDataMethods.MLP import MLP
from Methods.TabularDataMethods.RandomFeature.RFF import RFFDerivativeProviderST

def require_evaluate(provider):
    if not hasattr(provider, "evaluate") or not callable(provider.evaluate):
        raise TypeError(
            f"Provider must implement an evaluate(s_q, T_q) method "
            "returning (q, dq_ds, dq_dT)."
        )

def scaled_flux(provider, s_mean, s_std, T_mean, T_std):
    require_evaluate(provider)
    def flux_law(s, T, xg):
        s_hat = (s - s_mean) / s_std
        T_hat = (T - T_mean) / T_std
        q, dq_ds_hat, dq_dT_hat = provider.evaluate(np.array([s_hat]), np.array([T_hat]))
        return (float(q[0]), float(dq_ds_hat[0] / s_std), float(dq_dT_hat[0] / T_std))
    return flux_law

def unscaled_flux(provider):
    require_evaluate(provider)
    def flux_law(s, T, xg):
        q, dq_ds, dq_dT = provider.evaluate(np.array([s]), np.array([T]))
        return (float(q[0]), float(dq_ds[0]), float(dq_dT[0]))
    return flux_law

def build_provider(method, df, training_df):
    method_key = method.lower()
    start = time.perf_counter()

    k0 = float(df["k_0"].iloc[0])
    alpha = float(df["alpha"].iloc[0])
    beta = float(df["beta"].iloc[0])

    s_mean = float(training_df["s"].mean())
    s_std = float(training_df["s"].std()) or 1.0
    T_mean = float(training_df["T"].mean())
    T_std = float(training_df["T"].std()) or 1.0

    s_grid, T_grid, q_grid = structured_table_from_chopped_data(training_df)
    h_s = float(np.median(np.diff(s_grid)))
    h_T = float(np.median(np.diff(T_grid)))

    s_hat_grid, T_hat_grid, q_grid = scaled_structured_table_from_physical(
        s_grid, T_grid, q_grid, s_mean, s_std, T_mean, T_std
    )
    s_hat_data = (training_df["s"].to_numpy() - s_mean) / s_std
    T_hat_data = (training_df["T"].to_numpy() - T_mean) / T_std
    q_noisy_data = training_df["q_noisy"].to_numpy()

    if method_key == "analytic":
        def flux_law(s, T, xg):
            return (
                float(q_true(s, T, k0, alpha, beta)),
                float(a_true(s, T, k0, alpha, beta)),
                float(b_true(s, T, k0, alpha, beta)),
            )

    elif method_key == "cubicspline":
        provider = CubicSplineFluxST(s_hat_grid, T_hat_grid, q_grid)
        flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)

    elif method_key == "pchip":
        q_grid_pchip = project_table_nonincreasing_in_s(q_grid)

        provider = PchipFluxST(
            s_hat_grid, T_hat_grid, q_grid_pchip,
            extrapolate=False,
            clip=True,
        )
        flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)

    elif method_key in {"smooth+pchip", "smoothpchip"}:
        q_grid_smooth = smooth_table_3x3(q_grid)
        q_grid_smooth_pchip = project_table_nonincreasing_in_s(q_grid_smooth)

        provider = PchipFluxST(
            s_hat_grid, T_hat_grid, q_grid_smooth_pchip,
            extrapolate=False,
            clip=True,
        )
        flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)

    elif method_key == "rbf":
        provider = RBFDerivativeProviderST(
            s_hat_data, T_hat_data, q_noisy_data,
            function="gaussian",
            epsilon=1.1,
            smooth=1.0,
        )
        flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)

    elif method_key == "savgol":
        provider = LocalSavGolProvider(
            np.column_stack([s_hat_data, T_hat_data]),
            q_noisy_data,
            K=min(SAVGOL_NEIGHBORS, len(training_df)),
            h=0.9,
        )
        flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)

    elif method_key in {"kiss-gp", "kissgp"}:
        gp_subset = training_df.sample(n=min(250, len(training_df)), random_state=4)

        # FIXME: this maybe should use scaled inputs / scaled_flux()?

        provider = KISSGPFluxST(
            gp_subset["s"].to_numpy(),
            gp_subset["T"].to_numpy(),
            gp_subset["q_noisy"].to_numpy(),
            grid_size=16,
            training_iter=20,
            learning_rate=0.08,
        )
        flux_law = unscaled_flux(provider)

    elif method_key == "finitediff":
        provider = TabularFiniteDifferenceProvider(s_grid, T_grid,q_grid)
        flux_law = unscaled_flux(provider)
        h_s = provider.ds
        h_T = provider.dT

    elif method_key == "mlp":
        provider = MLP(
            hidden_layer_sizes=(128, 128, 64, 64, 64, 64),
            scale_inputs=False, 
            num_epochs = 200, 
            l2_weight=0.0,           # set nonzero for regularization
            monotonicity_weight=0.0, # set nonzero for (closer to) monotone output
            verbose=False)

        x_train = np.column_stack([s_hat_data, T_hat_data])
        y_train = q_noisy_data
        provider.fit(x_train, y_train)

        flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)


    elif method_key == "rff":
        provider = RFFDerivativeProviderST(
            s_hat_data,
            T_hat_data,
            q_noisy_data,
            n_components=2000,
            gamma=1.0,
            alpha=1e-6,
            random_state=0,
        )

        flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)

        

    # ADD MORE METHODS/MODELS HERE


    else:
        raise ValueError(f"unknown method: {method}")

    return {
        "method": method,
        "flux": flux_law,
        "build_s": time.perf_counter() - start,
        "h_s": h_s,
        "h_T": h_T,
        "s_mean": s_mean,
        "s_std": s_std,
        "T_mean": T_mean,
        "T_std": T_std,
    }