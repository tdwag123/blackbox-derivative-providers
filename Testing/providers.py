import sys
from pathlib import Path
import time
import importlib.util

import numpy as np
from scipy.sparse import eye, lil_matrix
from scipy.sparse.linalg import spsolve

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
from Methods.TabularDataMethods.FDIntegratedEpanechnikov import (
    TabularFDIntegratedEpanechnikovSmoothST,
)
from Methods.TabularDataMethods.FDMaternSmoothing import TabularFDMaternSmoothST
from Methods.TabularDataMethods.MonotoneInterpolation import PchipFluxST
from Methods.TabularDataMethods.RadialBasisFunctions import RBFDerivativeProviderST
try:
    from Methods.TabularDataMethods.GaussianProcessesWrap import KISSGPFluxST
except (ImportError, OSError):
    KISSGPFluxST = None

try:
    from Methods.TabularDataMethods.MLP import MLP
except (ImportError, OSError):
    MLP = None

try:
    from Methods.TabularDataMethods.RandomFeature.RFF_ridge import RidgeRFFDerivativeProviderST
except (ImportError, OSError):
    RidgeRFFDerivativeProviderST = None

try:
    from Methods.TabularDataMethods.RandomFeature.RFF_penalty import PenalizedRFFDerivativeProviderST
except (ImportError, OSError):
    PenalizedRFFDerivativeProviderST = None


try:
    from Methods.TabularDataMethods.RandomFeature.RFF_lin_reg import LinRegRFFDerivativeProviderST
except (ImportError, OSError):
    LinRegRFFDerivativeProviderST = None

try:
    from Methods.TabularDataMethods.RandomFeature.RFF_general import RFFDerivativeProviderST
except (ImportError, OSError):
    RFFDerivativeProviderST = None

try:
    monotone_gp_path = (
        ROOT
        / "Methods"
        / "TabularDataMethods"
        / "MaternGP+Monotonicity"
        / "maternGPMonotoneRegUpdated.py"
    )
    monotone_gp_spec = importlib.util.spec_from_file_location(
        "maternGPMonotoneRegUpdated",
        monotone_gp_path,
    )
    monotone_gp_module = importlib.util.module_from_spec(monotone_gp_spec)
    sys.modules[monotone_gp_spec.name] = monotone_gp_module
    monotone_gp_spec.loader.exec_module(monotone_gp_module)
    MonotoneGPFluxST = monotone_gp_module.MonotoneGPFluxST
except (ImportError, OSError, AttributeError):
    MonotoneGPFluxST = None

def parse_method_spec(method):
    method_text = str(method)
    if "+" not in method_text:
        return method_text.lower(), None

    parts = method_text.split("+")
    base_parts = []
    regularization = None

    for option in parts:
        option_key = option.strip().lower()
        if option_key.startswith("reg="):
            regularization = parse_regularization_spec(option_key[4:])
        elif option_key.startswith("tikhonov="):
            regularization = parse_regularization_spec("laplacian:" + option_key[9:])
        elif option_key.startswith(("laplacian:", "gradient:", "grad:", "tikhonov:")):
            regularization = parse_regularization_spec(option_key)
        else:
            base_parts.append(option)

    return "+".join(base_parts).lower(), regularization


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
        s_data = training_df["s"].to_numpy(dtype=float)
        T_data = training_df["T"].to_numpy(dtype=float)
    else:
        S, T = np.meshgrid(s_grid, T_grid, indexing="ij")
        S_hat, T_hat = np.meshgrid(s_hat_grid, T_hat_grid, indexing="ij")
        s_hat_data = S_hat.ravel()
        T_hat_data = T_hat.ravel()
        q_noisy_data = q_grid.ravel()
        s_data = S.ravel()
        T_data = T.ravel()

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
        )
        flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)

    elif method_key in {"smooth+pchip", "smoothpchip"}:
        q_grid_smooth = smooth_table_3x3(q_grid)
        q_grid_smooth_pchip = project_table_nonincreasing_in_s(q_grid_smooth)

        provider = PchipFluxST(
            s_hat_grid, T_hat_grid, q_grid_smooth_pchip,
            extrapolate=False,
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


    elif method_key == "ridge_rff":
        if RidgeRFFDerivativeProviderST is None:
            raise ImportError("RFF requires scikit-learn dependencies")

        provider = RidgeRFFDerivativeProviderST(
            s_hat_data,
            T_hat_data,
            q_noisy_data,
            n_components=2000,
            gamma=1.0,
            alpha=1e-6,
            random_state=0,
        )

        flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)

    elif method_key == "penalty_rff":
        if PenalizedRFFDerivativeProviderST is None:
            raise ImportError("penalty_rff requires scikit-learn dependencies")

        provider = PenalizedRFFDerivativeProviderST(
            s_hat_data,
            T_hat_data,
            q_noisy_data,
            n_components=2000,
            gamma=1.0,
            alpha=1e-6,
            random_state=0,
        )

        flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)


    elif method_key == "lin_reg_rff":
        if LinRegRFFDerivativeProviderST is None:
            raise ImportError("lin_reg_rff requires scikit-learn dependencies")

        provider = LinRegRFFDerivativeProviderST(
            s_hat_data,
            T_hat_data,
            q_noisy_data,
            n_components=2000,
            gamma=1.0,
            alpha=1e-6,
            random_state=0,
        )

        flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)


    elif method_key == "rff":
        provider = RFFDerivativeProviderST(
            s_hat_data,
            T_hat_data,
            q_noisy_data,
            regularization='ridge',
            n_components=2000,
            gamma=1.0,
            alpha=1e-6,
            random_state=0,
        )

        flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)


    elif method_key in {"materngpmonotone", "materngpmonotone_unregularized", "materngpmonotone_regularized"}:
        if MonotoneGPFluxST is None:
            raise ImportError("maternGPMonotone requires GP dependencies")

        noise_std = (
            training_df["sigma"].to_numpy(dtype=float)
            if "sigma" in training_df.columns
            else None
        )
        use_internal_tikhonov = method_key == "materngpmonotone_regularized"

        provider = MonotoneGPFluxST(
            s_data,
            T_data,
            q_noisy_data,
            noise_std=noise_std,
            learn_neg_flux=True,
            n_virtual_per_axis=10,
            probit_nu=1e-3,
            ep_max_iter=20,
            ep_damping=0.7,
            ep_tol=1e-5,
            n_restarts_optimizer=0,
            random_state=42,
            use_tikhonov=use_internal_tikhonov,
            tikhonov_strength=1e-2,
            tikhonov_target="deriv",
            verbose=False,
        )

        # MonotoneGPFluxST.evaluate() accepts physical s and T and returns
        # physical q, dq/ds, and dq/dT, so do not use scaled_flux here.
        flux_law = unscaled_flux(provider)

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
