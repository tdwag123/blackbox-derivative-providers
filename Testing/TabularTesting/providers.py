import sys
from pathlib import Path
import time

import numpy as np
from scipy.sparse import eye, lil_matrix
from scipy.sparse.linalg import spsolve

from Testing.Deprecated.data import structured_table_from_chopped_data, scaled_structured_table_from_physical
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
from Methods.TabularDataMethods.FDIntegratedEpanechnikov import (
    TabularFDIntegratedEpanechnikovSmoothST,
)
from Methods.TabularDataMethods.FDMaternSmoothing import TabularFDMaternSmoothST
from Methods.TabularDataMethods.MonotoneInterpolation import PchipFluxST
from Methods.TabularDataMethods.RadialBasisFunctions import RBFDerivativeProviderST
from Methods.TabularDataMethods.RandomFeature.RFF import RFFDerivativeProviderST

try:
    from Methods.TabularDataMethods.GaussianProcessesWrap import KISSGPFluxST
except ModuleNotFoundError:
    KISSGPFluxST = None

try:
    from Methods.TabularDataMethods.MLP import MLP
except ModuleNotFoundError:
    MLP = None

# Adds the option to specify regularization type and strength in the method string, e.g.:
#   "cubicspline+reg=laplacian:0.1"
def parse_method_spec(method):
    method_text = str(method)
    if "+" not in method_text:
        return method_text.lower(), None

    base_method, *options = method_text.split("+")
    regularization = None

    for option in options:
        option_key = option.strip().lower()
        if option_key.startswith("reg="):
            regularization = parse_regularization_spec(option_key[4:])
        elif option_key.startswith("tikhonov="):
            regularization = parse_regularization_spec("laplacian:" + option_key[9:])
        elif option_key.startswith(("laplacian:", "gradient:", "grad:", "tikhonov:")):
            regularization = parse_regularization_spec(option_key)
        else:
            raise ValueError(f"unknown method option in {method_text!r}: {option}")

    return base_method.lower(), regularization

# Parses a regularization specification string
def parse_regularization_spec(spec):
    parts = spec.split(":")
    reg_type = parts[0].strip().lower()
    if reg_type in {"tikhonov", "curvature"}:
        reg_type = "laplacian"
    if reg_type == "grad":
        reg_type = "gradient"
    if reg_type not in {"laplacian", "gradient"}:
        raise ValueError(f"unknown regularization type: {reg_type}")

    strength = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
    return {"type": reg_type, "strength": strength}

# Smooths a 2D grid of values using Tikhonov regularization with the specified type and strength.
def tikhonov_regularize_grid(q_grid, reg_type, strength):
    q_grid = np.asarray(q_grid, dtype=float)
    if strength <= 0.0:
        return q_grid.copy()

    n_s, n_T = q_grid.shape
    n_unknowns = n_s * n_T

    def flat(i, j):
        return i * n_T + j

    rows = []
    if reg_type == "gradient":
        for i in range(n_s - 1):
            for j in range(n_T):
                rows.append({flat(i + 1, j): 1.0, flat(i, j): -1.0})
        for i in range(n_s):
            for j in range(n_T - 1):
                rows.append({flat(i, j + 1): 1.0, flat(i, j): -1.0})

    elif reg_type == "laplacian":
        for i in range(1, n_s - 1):
            for j in range(1, n_T - 1):
                rows.append({
                    flat(i, j): -4.0,
                    flat(i - 1, j): 1.0,
                    flat(i + 1, j): 1.0,
                    flat(i, j - 1): 1.0,
                    flat(i, j + 1): 1.0,
                })

    if not rows:
        return q_grid.copy()

    penalty = lil_matrix((len(rows), n_unknowns), dtype=float)
    for row_idx, row in enumerate(rows):
        for col_idx, value in row.items():
            penalty[row_idx, col_idx] = value

    penalty = penalty.tocsr()
    system = eye(n_unknowns, format="csr") + strength * (penalty.T @ penalty)
    rhs = q_grid.ravel()
    return spsolve(system, rhs).reshape(q_grid.shape)


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
    method_key, regularization = parse_method_spec(method)
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

    # check if regularization is specified
    if regularization is not None:
        q_grid = tikhonov_regularize_grid(
            q_grid,
            regularization["type"],
            regularization["strength"],
        )

    s_hat_grid, T_hat_grid, q_grid = scaled_structured_table_from_physical(
        s_grid, T_grid, q_grid, s_mean, s_std, T_mean, T_std
    )
    if regularization is None:
        s_hat_data = (training_df["s"].to_numpy() - s_mean) / s_std
        T_hat_data = (training_df["T"].to_numpy() - T_mean) / T_std
        q_noisy_data = training_df["q_noisy"].to_numpy()
    else:
        S_hat, T_hat = np.meshgrid(s_hat_grid, T_hat_grid, indexing="ij")
        s_hat_data = S_hat.ravel()
        T_hat_data = T_hat.ravel()
        q_noisy_data = q_grid.ravel()

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
        if KISSGPFluxST is None:
            raise ImportError("KISS-GP requires torch/gpytorch dependencies")

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

    elif method_key in {"fdmatern52", "fd-matern52", "fdmatern"}:
        provider = TabularFDMaternSmoothST(s_grid, T_grid, q_grid)
        flux_law = unscaled_flux(provider)
        h_s = provider.ds
        h_T = provider.dT

    elif method_key in {"fdintegratedepanechnikov", "fd-integrated-epanechnikov"}:
        provider = TabularFDIntegratedEpanechnikovSmoothST(s_grid, T_grid, q_grid)
        flux_law = unscaled_flux(provider)
        h_s = provider.ds
        h_T = provider.dT

    elif method_key == "mlp":
        if MLP is None:
            raise ImportError("MLP requires jax dependencies")

        provider = MLP(
            hidden_layer_sizes=(32, 16),
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
        "regularization_type": (
            regularization["type"] if regularization is not None else "none"
        ),
        "regularization_strength": (
            regularization["strength"] if regularization is not None else 0.0
        ),
        "s_mean": s_mean,
        "s_std": s_std,
        "T_mean": T_mean,
        "T_std": T_std,
    }
