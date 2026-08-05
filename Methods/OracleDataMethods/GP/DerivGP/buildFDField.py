"""
Builds centered finite-difference observations of dq/ds and dq/dT from noisy flux
oracle. 

For a local anchor x_a = (s_a, T_a), derivative centers z_i=(s_i, T_i) are 
placed inside the rectangle [s_a-R_s, s_a+R_s] x [T_a-R_T, T_a+R_T]. At each center, 
oracle is sampled at the four stencil points: 

    (s_i-h_s, T_i), (s_i+h, T_i),
    (s_i, T_i-h_T), (s_i, T_i+h_T),

and second-order centered differences

    q_s(z_i) ~= [q(s_i+h_s,T_i)-q(s_i-h_s,T_i)]/(2h_s),
    q_T(z_i) ~= [q(s_i,T_i+h_T)-q(s_i,T_i-h_T)]/(2h_T)

are formed. 

If y is the vector of (averaged (*)) oracle values at all unique cached stencil points and 
D = [D_s; D_T] is the stacked finite-difference operator, then d = Dy, where d = [q_s; q_T].
If the cached oracle means have covariance Sigma_q, the full derivative-noise covariance is 
propagated exactly as Sigma_d = D Sigma_q D^T. This retains correlations between derivative 
estimates that reuse the same oracle points.  
    (*) RMK: Averaging is only applicable when given a stochastic oracle. Set repeats=1
    in the present deterministic case. 

References:
1. More and Wild (2012); Estimating Derivatives of Noisy Simulations
2. Solak et al. (2003); Derivative Observations in Gaussian Process Models of
Dynamical Systems
"""

from pathlib import Path
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

def _call_oracle(oracle, s, T):
    """evaluate q(s,T) and return a floating-point np array"""
    value = oracle(s,T)
    return np.asarray(value, dtype=float)

def _learn_noise_variance(points, samples, jitter, n_restarts_optimizer):
    """estimate variance of one oracle call in physical flux units. prelim GP
    is fit to all repeated flux evals using k_total = k_Matern-5/2 + sigma_n^2 delta_ij.
    WhiteKernel parameter sigma_n^2 learned in standardized output units. that is, if 
    y_std = (y-y_mean)/y_scale, then Var(y_physical noise) = sigma_n,std^2 * y_scale^2.
    GaussianProcessRegressor(alpha=jitter) uses alpha only as numerical stabilization. 
    statistical observation noise is learned by WhiteKernel."""
    repeats, n_points = samples.shape
    X_raw = np.tile(points, (repeats, 1))
    y_raw = samples.reshape(-1)
    x_mean = X_raw.mean(axis=0)
    x_scale = X_raw.std(axis=0)
    x_scale[x_scale == 0.0] = 1.0
    y_mean = float(y_raw.mean())
    y_scale = float(y_raw.std())
    X = (X_raw - x_mean)/x_scale
    y = (y_raw - y_mean)/y_scale
    signal_kernel = ConstantKernel(1.0, (1e-4, 1e4)) * Matern(
        length_scale=np.ones(2),
        length_scale_bounds=(1e-2,1e2),
        nu=2.5)
    kernel = signal_kernel + WhiteKernel(
        noise_level=1e-3, 
        noise_level_bounds=(1e-10, 1e1))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=jitter,
        normalize_y=False,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=0)
    gp.fit(X,y)
    learned_variance_standardized = float(gp.kernel_.k2.noise_level)
    learned_variance_physical = learned_variance_standardized * y_scale**2
    return learned_variance_physical, learned_variance_standardized, str(gp.kernel_)

def _local_axis(center, radius, step, n):
    """construct derivative centers inside one local coordinate interval. step-sized 
    margin ensures both centered-difference endpoints remain inside [center-rad, center+rad]"""
    return np.linspace(center - radius + step, center + radius - step, n)

def build_finite_difference_field(
    oracle, # callable returning physical flux q(s,T)
    anchor, # local quadrature state (s_a, T_a); defines center of sampling patch and flux anchor
    s_radius, # half width R_s of local sampling rectangle
    T_radius, # half width R_T of local sampling rectangle
    n_s = 7, # should be odd to inclde anchor among Cartesian centers
    n_T = 7, # should be odd to include anchor among Cartesian centers
    h_s = None, # centered finite difference half-width (default = R_s/n_s)
    h_T = None, # centered finite difference half-width (default = R_T/n_T)
    repeats = 1, # this is a parameter that can be increased in stochastic case 
    jitter = 1e-8, # small diagonal numerical stabilization used in prelim GP
    n_restarts_optimizer = 0, # increase if learned noise consistently doesn't match true
    noise_variance_floor = 1e-12, # lb on var assigned to each (averaged) cache val
    output_file = None, # optional npz destination
):
    """build local derivative pseudo-observations aroudn one FEM quadrature state."""
    anchor = np.asarray(anchor, dtype=float).reshape(-1)
    anchor = (float(anchor[0]), float(anchor[1]))
    s_anchor, T_anchor = anchor
    s_radius = float(s_radius)
    T_radius = float(T_radius)
    n_s = int(n_s)
    n_T = int(n_T)
    repeats = int(repeats) 
    if h_s is None:
        h_s = s_radius/n_s
    if h_T is None:
        h_T = T_radius/n_T
    h_s = float(h_s)
    h_T = float(h_T)
    s_axis = _local_axis(s_anchor, s_radius, h_s, n_s)
    T_axis = _local_axis(T_anchor, T_radius, h_T, n_T)
    S, TT = np.meshgrid(s_axis, T_axis, indexing="ij")
    centers = np.column_stack([S.ravel(), TT.ravel()])

    # cache unique stencil locations so repeated finite diff can reuse same oracle evals
    points = []
    point_index = {}

    def add_point(point):
        point = np.asarray(point, dtype=float).reshape(-1)
        stored_point = (float(point[0]), float(point[1]))
        # rounding provides a canonical key for nominally identical floating-point stencil locations
        key = (round(stored_point[0], 12), round(stored_point[1], 12))
        if key not in point_index:
            point_index[key] = len(points)
            points.append(stored_point)
        return point_index[key]
        
    n = centers.shape[0]
    left = np.empty(n, dtype=int)
    right = np.empty(n, dtype=int)
    down = np.empty(n, dtype=int)
    up = np.empty(n, dtype=int)
    for i, (s,T) in enumerate(centers):
        left[i] = add_point((s-h_s, T))
        right[i] = add_point((s+h_s, T))
        down[i] = add_point((s, T-h_T))
        up[i] = add_point((s, T+h_T))
    anchor_index = add_point(anchor)
    cache_points = np.asarray(points, dtype=float)

    # what follows is applicable to stochastic case, where we sample repeatedly then avg; can
    # also increase repeats if want to dampen deterministic noise
    samples = np.empty((repeats, cache_points.shape[0]))
    for repeat in range(repeats):
        samples[repeat] = _call_oracle(oracle, cache_points[:, 0], cache_points[:, 1]).reshape(-1)
    q_mean = samples.mean(axis=0)
    learned_noise_variance, learned_noise_variance_standardized, noise_gp_kernel = (
        _learn_noise_variance(
            cache_points, 
            samples, 
            jitter, 
            n_restarts_optimizer,
        )
    )

    # if one oracle call has variance sigma_q^2, average of repeats independent
    # calls has variance sigma_q^2/repeats
    q_mean_variance = max(learned_noise_variance/repeats, noise_variance_floor)
    cache_q_variance = np.full(cache_points.shape[0], q_mean_variance)
    # finite diff matrices; dense arrays retained because local fields are small
    D_s = np.zeros((n, cache_points.shape[0]))
    D_T = np.zeros((n, cache_points.shape[0]))
    rows = np.arange(n)
    D_s[rows, left] = -1.0 / (2.0 * h_s)
    D_s[rows, right] = 1.0 / (2.0 * h_s)
    D_T[rows, down] = -1.0 / (2.0 * h_T)
    D_T[rows, up] = 1.0 / (2.0 * h_T)
    dq_ds = D_s @ q_mean
    dq_dT = D_T @ q_mean
    # with Sigma_q = diag(cache_q_variance), propagate averaged-oracle noise through
    # linear finite-diff map Sigma_d = D Sigma_q D^T; off diag terms remain if two deriv 
    # stencils share a cached oracle point.
    D = np.vstack([D_s, D_T])
    derivative_noise_covariance = (D * cache_q_variance[None, :]) @ D.T
    derivative_noise_covariance = 0.5 * (derivative_noise_covariance + derivative_noise_covariance.T)

    field = {
        "X_dq_ds": centers,
        "dq_ds": dq_ds,
        "X_dq_dT": centers.copy(),
        "dq_dT": dq_dT,
        "derivative_noise_covariance": derivative_noise_covariance,
        "anchor_point": anchor,
        "anchor_q": float(q_mean[anchor_index]),
        "anchor_q_variance": float(cache_q_variance[anchor_index]),
        "sampling_radii": np.array([s_radius, T_radius]),
        "local_bounds": np.array([[s_anchor - s_radius, s_anchor + s_radius], [T_anchor - T_radius, T_anchor + T_radius]]),
        "cache_points": cache_points,
        "cache_q": q_mean,
        "cache_q_variance": cache_q_variance,
        "learned_oracle_noise_variance": float(learned_noise_variance),
        "learned_oracle_noise_std": float(np.sqrt(learned_noise_variance)),
        "learned_oracle_noise_variance_standardized": float(learned_noise_variance_standardized),
        "noise_gp_kernel": noise_gp_kernel,
        "h_s": h_s,
        "h_T": h_T,
        "repeats": repeats,
        "oracle_calls": repeats * cache_points.shape[0],
    }

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_file, **field)

    return field

"""smoke test below; check if generates an appropriate deriv field"""
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    hidden_noise_std = 0.05

    def oracle(s, T):
        q = -(1.25 * (1.0 + 0.18 * T**2) + 0.12 * s**2) * s
        return q + rng.normal(0.0, hidden_noise_std, size=np.shape(q))

    quadrature_state = (0.35, 1.4)
    output = Path(__file__).resolve().parent / "fd_derivative_field_local.npz"

    field = build_finite_difference_field(
        oracle,
        anchor=quadrature_state,
        s_radius=0.45,
        T_radius=0.30,
        n_s=5,
        n_T=5,
        repeats=4,
        output_file=output,
    )

    print("Anchor:", field["anchor_point"])
    print("Local bounds:\n", field["local_bounds"])
    print("Derivative points:", len(field["dq_ds"]))
    print("Oracle calls:", field["oracle_calls"])
    print("Learned one-call noise std:", field["learned_oracle_noise_std"])
    print("Saved:", output)