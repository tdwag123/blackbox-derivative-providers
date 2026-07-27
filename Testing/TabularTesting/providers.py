import sys
from pathlib import Path
import time
import importlib.util

import numpy as np
from scipy.sparse import eye, lil_matrix
from scipy.sparse.linalg import spsolve

from Testing.Deprecated.data import structured_table_from_chopped_data, scaled_structured_table_from_physical
# from tabular_models import (
#     LocalSavGolProvider,
#     SAVGOL_NEIGHBORS,
#     TabularFiniteDifferenceProvider,
#     project_table_nonincreasing_in_s,
#     smooth_table_3x3,
# )

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
from Methods.TabularDataMethods.KernelMethods import KernelDerivativeProviderST
try:
    from Methods.TabularDataMethods.GaussianProcessesWrap import KISSGPFluxST
except (ImportError, OSError):
    KISSGPFluxST = None

try:
    from Methods.TabularDataMethods.MLP import MLP
except (ImportError, OSError):
    MLP = None

try:
    from Methods.TabularDataMethods.RandomFeature.unconstrained_regularized_least_squares_buildup.RFF_ridge import RidgeRFFDerivativeProviderST
except (ImportError, OSError):
    RidgeRFFDerivativeProviderST = None

try:
    from Methods.TabularDataMethods.RandomFeature.unconstrained_regularized_least_squares_buildup.RFF_penalty import PenalizedRFFDerivativeProviderST
except (ImportError, OSError):
    PenalizedRFFDerivativeProviderST = None

try:
    from Methods.TabularDataMethods.RandomFeature.unconstrained_regularized_least_squares_buildup.RFF_lin_reg import LinRegRFFDerivativeProviderST
except (ImportError, OSError):
    LinRegRFFDerivativeProviderST = None

try:
    from Methods.TabularDataMethods.RandomFeature.unconstrained_regularized_least_squares_buildup.RFF_general import RFFDerivativeProviderST
except (ImportError, OSError):
    RFFDerivativeProviderST = None

try:
    from Methods.TabularDataMethods.RandomFeature.RFF_final import FlexRFFDerivativeProviderST
except (ImportError, OSError):
    FlexRFFDerivativeProviderST = None

try:
    from Methods.TabularDataMethods.RandomFeature.RFF_constrained_final import ConstrainedRFFDerivativeProviderST
except (ImportError, OSError):
    ConstrainedRFFDerivativeProviderST = None

try:
    from Methods.TabularDataMethods.MaternGPMonotonicity.maternGPMonotone import MonotoneGPFluxST
except (ImportError, OSError, AttributeError):
    MonotoneGPFluxST = None

try:
    from Methods.TabularDataMethods.monotoneGP_KPEP.monotoneGPKPEP import MonotoneGPKPFluxST
except (ImportError, OSError):
     MonotoneGPKPFluxST = None 
    
def parse_method_spec(method):
    method_text = str(method)
    if "+" not in method_text:
        return method_text.lower(), None, {}

    parts = method_text.split("+")
    base_parts = []
    regularization = None
    options = {}

    for option in parts:
        option_key = option.strip().lower()
        if option_key.startswith("reg="):
            regularization = parse_regularization_spec(option_key[4:])
        elif option_key.startswith("tikhonov="):
            regularization = parse_regularization_spec("laplacian:" + option_key[9:])
        elif option_key.startswith(("laplacian:", "gradient:", "grad:", "tikhonov:")):
            regularization = parse_regularization_spec(option_key)
        elif option_key.startswith("ridge="):
            options["ridge_strength"] = float(option_key[6:])
        elif option_key.startswith("ridge:"):
            options["ridge_strength"] = float(option_key[6:])
        elif option_key.startswith("alpha="):
            options["ridge_strength"] = float(option_key[6:])
        elif option_key.startswith("alpha:"):
            options["ridge_strength"] = float(option_key[6:])
        elif option_key.startswith("epsilon="):
            options["epsilon"] = float(option_key[8:])
        elif option_key.startswith("epsilon:"):
            options["epsilon"] = float(option_key[8:])
        elif option_key.startswith("eps="):
            options["epsilon"] = float(option_key[4:])
        elif option_key.startswith("eps:"):
            options["epsilon"] = float(option_key[4:])
        elif option_key.startswith("length_scale="):
            options["epsilon"] = float(option_key[13:])
        elif option_key.startswith("length_scale:"):
            options["epsilon"] = float(option_key[13:])
        else:
            base_parts.append(option)

    return "+".join(base_parts).lower(), regularization, options


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

    provider = None
    method_key, regularization, method_options = parse_method_spec(method)
    model_regularization_type = ("ridge" if method_options.get("ridge_strength", 0.0) > 0.0 else "none")
    model_regularization_strength = method_options.get("ridge_strength", 0.0)
    start = time.perf_counter()
    provider = None

    k0 = float(df["k_0"].iloc[0])
    alpha = float(df["alpha"].iloc[0])
    beta = float(df["beta"].iloc[0])

    s_mean = float(training_df["s"].mean())
    s_std = float(training_df["s"].std()) or 1.0
    T_mean = float(training_df["T"].mean())
    T_std = float(training_df["T"].std()) or 1.0

    s_grid, T_grid, q_grid_physical, observation_mask, sigma_grid = structured_table_from_chopped_data(training_df)

    q_grid_kp = q_grid_physical.copy()

    h_s = float(np.median(np.diff(s_grid)))
    h_T = float(np.median(np.diff(T_grid)))

    if regularization is not None:
        q_grid_physical = tikhonov_regularize_grid(
            q_grid_physical,
            regularization["type"],
            regularization["strength"],
        )

    s_hat_grid, T_hat_grid, q_grid_scaled = scaled_structured_table_from_physical(
        s_grid, T_grid, q_grid_physical, s_mean, s_std, T_mean, T_std
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
        q_noisy_data = q_grid_scaled.ravel()
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
        provider = CubicSplineFluxST(s_hat_grid, T_hat_grid, q_grid_scaled)
        flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)

    # elif method_key == "pchip":
    #     q_grid_pchip = project_table_nonincreasing_in_s(q_grid_scaled)

    #     provider = PchipFluxST(
    #         s_hat_grid, T_hat_grid, q_grid_pchip,
    #         extrapolate=False,
    #     )
    #     flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)

    # elif method_key in {"smooth+pchip", "smoothpchip"}:
    #     q_grid_smooth = smooth_table_3x3(q_grid_scaled)
    #     q_grid_smooth_pchip = project_table_nonincreasing_in_s(q_grid_smooth)

    #     provider = PchipFluxST(
    #         s_hat_grid, T_hat_grid, q_grid_smooth_pchip,
    #         extrapolate=False,
    #     )
    #     flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)

    elif method_key == "rbf":
        method_options.setdefault("epsilon", 2.3)
        method_options.setdefault("ridge_strength", 1.0e-3)
        provider = KernelDerivativeProviderST(
            s_hat_data, T_hat_data, q_noisy_data,
            function="gaussian",
            epsilon=method_options["epsilon"],
            smooth=0.0,
            ridge_strength=method_options["ridge_strength"],
        )
        flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)

    elif method_key in {"matern52_krr", "matern52-krr", "matern_krr", "matern-krr"}:
        method_options.setdefault("epsilon", 4.8)
        method_options.setdefault("ridge_strength", 1.0e-4)
        provider = KernelDerivativeProviderST(
            s_hat_data,
            T_hat_data,
            q_noisy_data,
            function="matern52",
            epsilon=method_options["epsilon"],
            smooth=0.0,
            ridge_strength=method_options["ridge_strength"],
        )
        flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)

    # elif method_key == "savgol":
    #     provider = LocalSavGolProvider(
    #         np.column_stack([s_hat_data, T_hat_data]),
    #         q_noisy_data,
    #         K=min(SAVGOL_NEIGHBORS, len(training_df)),
    #         h=0.9,
    #     )
    #     flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)

    elif method_key in {"kiss-gp", "kissgp"}:
        if KISSGPFluxST is None:
            raise ImportError("KISS-GP requires torch/gpytorch dependencies")

        rng = np.random.default_rng(4)
        gp_indices = np.arange(len(q_noisy_data))
        if len(gp_indices) > 250:
            gp_indices = rng.choice(gp_indices, size=250, replace=False)

        # FIXME: this maybe should use scaled inputs / scaled_flux()?

        provider = KISSGPFluxST(
            s_data[gp_indices],
            T_data[gp_indices],
            q_noisy_data[gp_indices],
            grid_size=16,
            training_iter=20,
            learning_rate=0.08,
            ridge_strength=method_options.get("ridge_strength", 0.0),
        )
        flux_law = unscaled_flux(provider)

    # elif method_key == "finitediff":
    #     provider = TabularFiniteDifferenceProvider(s_grid, T_grid,q_grid_physical)
    #     flux_law = unscaled_flux(provider)
    #     h_s = provider.ds
    #     h_T = provider.dT

    elif method_key in {"fdmatern52", "fd-matern52", "fdmatern"}:
        provider = TabularFDMaternSmoothST(s_grid, T_grid, q_grid_physical)
        flux_law = unscaled_flux(provider)
        h_s = provider.ds
        h_T = provider.dT

    elif method_key in {"fdintegratedepanechnikov", "fd-integrated-epanechnikov"}:
        provider = TabularFDIntegratedEpanechnikovSmoothST(s_grid, T_grid, q_grid_physical)
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
            l2_weight=method_options.get("ridge_strength", 0.0),
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
            alpha=method_options.get("ridge_strength", 1e-6),
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
            alpha=method_options.get("ridge_strength", 1e-6),
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

    elif method_key == "rff_flexible+alpha:1e-6, p=2, n_components=2000":
        """
        alpha: regularization strength.
            if larger, smoother model + less overfitting + more possible underfitting. 
            if smaller, more flexible model but more risk of fitting noise.
            if 0, no regularization (=> LINEAR REGRESSION), regardless of p.
        freq_weight: frequency weighting, p.
            if p=0, no frequency weighting (=> RIDGE REGRESSION).
            if p=1: penalty grows linearly with frequency length.
            if p=2: penalty grows quadratically with frequency length.
            if p>2: high-frequency features are punished very strongly.
        """
        provider = FlexRFFDerivativeProviderST(
            s_hat_data,
            T_hat_data,
            q_noisy_data,
            alpha=1e-6,
            freq_weight=2,
            n_components=2000,
            gamma=1.0,
            random_state=0,
        )

        flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)


    elif method_key == "rff_flexible+alpha:1e-6, p=4, n_components=2000":
        """
        alpha: regularization strength.
            if larger, smoother model + less overfitting + more possible underfitting. 
            if smaller, more flexible model but more risk of fitting noise.
            if 0, no regularization (=> LINEAR REGRESSION), regardless of p.
        freq_weight: frequency weighting, p.
            if p=0, no frequency weighting (=> RIDGE REGRESSION).
            if p=1: penalty grows linearly with frequency length.
            if p=2: penalty grows quadratically with frequency length.
            if p>2: high-frequency features are punished very strongly.
        """
        provider = FlexRFFDerivativeProviderST(
            s_hat_data,
            T_hat_data,
            q_noisy_data,
            alpha=1e-6,
            freq_weight=4,
            n_components=2000,
            gamma=1.0,
            random_state=0,
        )

        flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)

    elif method_key == "rff_constrained, alpha=1e-6, p=2, n_components=2000, osqp":
        provider = ConstrainedRFFDerivativeProviderST(
            s_hat_data,
            T_hat_data,
            q_noisy_data,
            alpha=1e-6,
            freq_weight=2,
            n_components=2000,
            gamma=1.0,
            random_state=0,
        )

        flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)


    elif method_key == "rff_constrained, alpha=1e-6, p=4, n_components=2000, osqp":
        provider = ConstrainedRFFDerivativeProviderST(
            s_hat_data,
            T_hat_data,
            q_noisy_data,
            alpha=1e-6,
            freq_weight=4,
            n_components=2000,
            gamma=1.0,
            random_state=0,
        )

        flux_law = scaled_flux(provider, s_mean, s_std, T_mean, T_std)



    elif method_key in {"materngpmonotone", "materngpmonotone_unregularized", "materngpmonotone_regularized"}:
        if MonotoneGPFluxST is None:
            raise ImportError("maternGPMonotone requires NumPy, SciPy, and scikit-learn")
        
        regularized = (
            method_key == "materngpmonotone_regularized"
            or method_options.get("ridge_strength", 0.0) > 0.0
        )
        function_strength = 1e-4 if regularized else 0.0
        derivative_strength = (
            method_options.get("ridge_strength", 1e-2) if regularized else 0.0
        )

        relative_sigma = training_df["sigma"].to_numpy(dtype=float)
        noise_std = (relative_sigma * np.maximum(1.0, np.abs(q_noisy_data)))
        provider = MonotoneGPFluxST(
            s_train=s_data,
            T_train=T_data,
            q_train=q_noisy_data,
            noise_std=noise_std,
            learn_neg_flux=True,

            # Initial monotonicity grid: 6 x 6 = 36 points.
            n_virtual_per_axis=6,

            # Dense grid used to look for remaining sign violations.
            monotonicity_check_points_per_axis=25,

            # Add virtual points at detected violations and refit.
            max_virtual_refinements=3,
            max_virtual_points_per_round=16,

            # Probit approximation to df/ds >= 0.
            probit_nu=1e-4,

            # EP controls.
            ep_max_iter=100,
            ep_damping=0.3,
            ep_tol=1e-5,

            # Separate function and derivative regularization.
            function_regularization=function_strength,
            derivative_regularization=derivative_strength,

            # Numerical variance floor for exactly zero-noise data.
            minimum_noise_variance=1e-8,

            # Ordinary-GP hyperparameter fit.
            lengthscale_bounds=(0.05, 100.0),
            n_restarts_optimizer=0,

            # Newton should normally remain inside the training domain; for now we let it be true
            allow_extrapolation=True,
        )

        if regularized:
            model_regularization_type = ("function+derivative")
            model_regularization_strength = (f"function={function_strength:g}; derivative={derivative_strength:g}")

        # The provider receives physical s and T and returns physical
        # q, dq/ds, and dq/dT.
        flux_law = unscaled_flux(provider)
    
    elif method_key in {"monotonegpkpep", "monotonegpkpep_unregularized", "monotonegpkpep_regularized"}:
        if MonotoneGPKPFluxST is None:
            raise ImportError("MonotoneGPKPFluxST could not be imported")
        
        regularized = (
            method_key == "monotonegpkpep_regularized"
            or method_options.get("ridge_strength", 0.0) > 0.0
        )
        function_strength = 1.0e-5 if regularized else 0.0
        derivative_strength = (
            method_options.get("ridge_strength", 1.0e-5)
            if regularized 
            else 0.0)
        
        provider = build_monotone_kp_provider(MonotoneGPKPFluxST, s_grid, T_grid, q_grid_kp, 
                                              observation_mask, sigma_grid, use_tikhonov=regularized, function_regularization=function_strength,
                                              derivative_regularization=derivative_strength)
        flux_law = unscaled_flux(provider)

        if regularized:
            model_regularization_type="function+derivative"
            model_regularization_strength=(f"function={function_strength:g}; derivative={derivative_strength:g}")

    # ADD MORE METHODS/MODELS HERE

    else:
        raise ValueError(f"unknown method: {method}")

    return {
        "method": method,
        "flux": flux_law,
        "build_s": time.perf_counter() - start,
        "h_s": h_s,
        "h_T": h_T,
        "smoothing_type": (
            regularization["type"] if regularization is not None else "none"
        ),
        "smoothing_strength": (
            regularization["strength"] if regularization is not None else 0.0
        ),
        "model_regularization_type": (
            model_regularization_type
        ),
        "model_regularization_strength": (model_regularization_strength),
        "s_mean": s_mean,
        "s_std": s_std,
        "T_mean": T_mean,
        "T_std": T_std,
        "provider": provider,
    }

# !!only for GP monotone KPEP!!
def build_monotone_kp_provider(
    provider_class,
    s_grid,
    T_grid,
    q_grid_physical,
    observation_mask,
    sigma_grid,
    use_tikhonov=False,
    function_regularization=0.0,
    derivative_regularization=0.0,
):
    return provider_class(
        s_grid,
        T_grid,
        q_grid_physical,
        noise_std=sigma_grid,
        observation_mask=observation_mask,

        # sigma values from the CSV are relative
        noise_is_relative=True,

        learn_neg_flux=True,
        nu=2.5,

        lengthscale="auto",
        lengthscale_candidates=(
            0.25,
            0.5,
            1.0,
            2.0,
            4.0,
        ),

        variance=1.0,

        n_virtual_per_axis=15,
        probit_nu=1.0e-6,

        ep_max_iter=100,
        ep_damping=0.4,
        ep_tol=1.0e-6,

        jitter=1.0e-10,

        use_tikhonov=use_tikhonov,
        function_regularization=(
            function_regularization
        ),
        derivative_regularization=(
            derivative_regularization
        ),

        variance_batch_size=32,
        prediction_batch_size=4096,
        verbose=False,
    )