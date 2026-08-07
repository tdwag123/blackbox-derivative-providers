from pathlib import Path
import itertools
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

def _learn_noise_variance(points, samples, jitter, n_restarts_optimizer):
    repeats, n_points, d_q = samples.shape
    X_raw = np.tile(points, (repeats, 1))
    learned_physical = np.empty(d_q)
    learned_standardized = np.empty(d_q)
    kernels = []
    for r in range(d_q):
        y_raw = samples[:, :, r].reshape(-1)
        x_mean = X_raw.mean(axis=0)
        x_scale = X_raw.std(axis=0)
        x_scale[x_scale == 0.0] = 1.0
        y_mean = float(y_raw.mean())
        y_scale = float(y_raw.std()) or 1.0
        X = (X_raw - x_mean) / x_scale
        y = (y_raw - y_mean) / y_scale
        kernel = ConstantKernel(1.0, (1e-4, 1e4)) * Matern(
            length_scale=np.ones(points.shape[1]),
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        ) + WhiteKernel(
            noise_level=1e-3,
            noise_level_bounds=(1e-10, 1e1),
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=jitter,
            normalize_y=False,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=0,
        )
        gp.fit(X, y)
        var_std = float(gp.kernel_.k2.noise_level)
        learned_standardized[r] = var_std
        learned_physical[r] = var_std * y_scale**2
        kernels.append(str(gp.kernel_))
    return learned_physical, learned_standardized, np.asarray(kernels)

def build_finite_difference_field(
    oracle,
    anchor_s,
    anchor_T,
    *,
    s_radius,
    T_radius,
    n_s=3,
    n_T=3,
    h_s=None,
    h_T=None,
    repeats=1,
    jitter=1e-8,
    n_restarts_optimizer=0,
    noise_variance_floor=1e-12,
    output_file=None,
):
    anchor_s = np.asarray(anchor_s, dtype=float).reshape(-1)
    anchor_T = np.asarray(anchor_T, dtype=float).reshape(-1)
    d_s, d_T = anchor_s.size, anchor_T.size
    d_input = d_s + d_T
    s_radius = np.broadcast_to(np.asarray(s_radius, dtype=float).reshape(-1), (d_s,)).copy()
    T_radius = np.broadcast_to(np.asarray(T_radius, dtype=float).reshape(-1), (d_T,)).copy()
    n_s = np.broadcast_to(np.asarray(n_s, dtype=int).reshape(-1), (d_s,)).copy()
    n_T = np.broadcast_to(np.asarray(n_T, dtype=int).reshape(-1), (d_T,)).copy()
    h_s = s_radius / n_s if h_s is None else np.broadcast_to(np.asarray(h_s, dtype=float).reshape(-1), (d_s,)).copy()
    h_T = T_radius / n_T if h_T is None else np.broadcast_to(np.asarray(h_T, dtype=float).reshape(-1), (d_T,)).copy()
    anchor = np.concatenate([anchor_s, anchor_T])
    radii = np.concatenate([s_radius, T_radius])
    steps = np.concatenate([h_s, h_T])
    counts = np.concatenate([n_s, n_T])
    axes = [np.linspace(anchor[j] - radii[j] + steps[j], anchor[j] + radii[j] - steps[j], int(counts[j]))
            for j in range(d_input)]
    centers = np.asarray(list(itertools.product(*axes)), dtype=float)
    n_centers = centers.shape[0]
    points, point_index = [], {}
    def add_point(point):
        point = np.asarray(point, dtype=float).reshape(-1)
        key = tuple(np.round(point, 12))
        if key not in point_index:
            point_index[key] = len(points)
            points.append(tuple(float(v) for v in point))
        return point_index[key]
    minus = np.empty((d_input, n_centers), dtype=int)
    plus = np.empty((d_input, n_centers), dtype=int)
    for j in range(d_input):
        for i, center in enumerate(centers):
            left = center.copy(); left[j] -= steps[j]
            right = center.copy(); right[j] += steps[j]
            minus[j, i] = add_point(left)
            plus[j, i] = add_point(right)
    anchor_index = add_point(anchor)
    cache_points = np.asarray(points, dtype=float)
    probe = np.asarray(oracle(cache_points[:1, :d_s], cache_points[:1, d_s:]), dtype=float).reshape(1, -1)
    d_q = probe.shape[1]
    samples = np.empty((int(repeats), cache_points.shape[0], d_q))
    for rep in range(int(repeats)):
        values = np.asarray(oracle(cache_points[:, :d_s], cache_points[:, d_s:]), dtype=float)
        samples[rep] = values.reshape(cache_points.shape[0], d_q)
    q_mean = samples.mean(axis=0)
    learned_var, learned_var_std, noise_kernels = _learn_noise_variance(cache_points, samples, jitter, n_restarts_optimizer)
    q_mean_var = np.maximum(learned_var / int(repeats), noise_variance_floor)
    cache_q_variance = np.broadcast_to(q_mean_var[None, :], q_mean.shape).copy()
    D_blocks = []
    derivatives = np.empty((n_centers, d_q, d_input))
    rows = np.arange(n_centers)
    for j in range(d_input):
        D_j = np.zeros((n_centers, cache_points.shape[0]))
        D_j[rows, minus[j]] = -1.0 / (2.0 * steps[j])
        D_j[rows, plus[j]] = 1.0 / (2.0 * steps[j])
        D_blocks.append(D_j)
        derivatives[:, :, j] = D_j @ q_mean
    D = np.vstack(D_blocks)
    derivative_noise_covariance = np.empty((d_q, D.shape[0], D.shape[0]))
    for r in range(d_q):
        cov = (D * cache_q_variance[:, r][None, :]) @ D.T
        derivative_noise_covariance[r] = 0.5 * (cov + cov.T)
    field = {
        "X_derivative": centers,
        "derivatives": derivatives,
        "derivative_noise_covariance": derivative_noise_covariance,
        "anchor_s": anchor_s,
        "anchor_T": anchor_T,
        "anchor_q": q_mean[anchor_index].copy(),
        "anchor_q_variance": cache_q_variance[anchor_index].copy(),
        "sampling_radii": radii,
        "local_bounds": np.column_stack([anchor - radii, anchor + radii]),
        "cache_points": cache_points,
        "cache_q": q_mean,
        "cache_q_variance": cache_q_variance,
        "learned_oracle_noise_variance": learned_var,
        "learned_oracle_noise_std": np.sqrt(learned_var),
        "learned_oracle_noise_variance_standardized": learned_var_std,
        "noise_gp_kernel": noise_kernels,
        "h": steps,
        "counts": counts,
        "d_s": d_s,
        "d_T": d_T,
        "d_q": d_q,
        "repeats": int(repeats),
        "oracle_calls": int(repeats) * cache_points.shape[0],
    }
    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_file, **field)
    return field