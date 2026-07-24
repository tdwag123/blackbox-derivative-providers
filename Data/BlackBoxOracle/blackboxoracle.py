import numpy as np


ORACLE_CONFIGS = {
    "nonlinear_high_noise": {
        "k_0": 1.0,
        "alpha": 0.5,
        "beta": 0.2,
        "sigma": 0.10,
    },
    "nonlinear_low_noise": {
        "k_0": 0.7,
        "alpha": 0.2,
        "beta": 0.05,
        "sigma": 0.02,
    },
    "linear_medium_noise": {
        "k_0": 1.5,
        "alpha": 0.0,
        "beta": 0.0,
        "sigma": 0.05,
    },
    "nonlinear_no_noise": {
        "k_0": 1.0,
        "alpha": 0.5,
        "beta": 0.2,
        "sigma": 0.0,
    },
    "linear_no_noise": {
        "k_0": 1.5,
        "alpha": 0.0,
        "beta": 0.0,
        "sigma": 0.0,
    },
}


def physical_flux(s, T, k_0, alpha, beta):
    return -(k_0 * (1.0 + alpha * T**2) + beta * s**2) * s


def physical_flux_derivatives(s, T, k_0, alpha, beta):
    q_s = -k_0 * (1.0 + alpha * T**2) - 3.0 * beta * s**2
    q_T = -2.0 * k_0 * alpha * T * s
    return q_s, q_T


def make_diffusion_oracle(config="nonlinear_high_noise", seed=None, noisy=True):
    if config not in ORACLE_CONFIGS:
        raise ValueError(f"Unknown oracle config: {config}")

    params = ORACLE_CONFIGS[config]
    rng = np.random.default_rng(seed)

    k_0 = params["k_0"]
    alpha = params["alpha"]
    beta = params["beta"]
    sigma = params["sigma"]

    def oracle(s, T, return_full=False):
        s = np.asarray(s, dtype=float)
        T = np.asarray(T, dtype=float)

        q_true = physical_flux(s, T, k_0, alpha, beta)
        q_s, q_T = physical_flux_derivatives(s, T, k_0, alpha, beta)

        if noisy:
            noise_std = sigma * np.maximum(1.0, np.abs(q_true))
            noise = noise_std * rng.standard_normal(np.shape(q_true))
            q = q_true + noise
        else:
            noise = np.zeros_like(q_true, dtype=float)
            q = q_true

        if return_full:
            return {
                "q": q,
                "q_true": q_true,
                "q_s": q_s,
                "q_T": q_T,
                "noise": noise,
                "k_0": k_0,
                "alpha": alpha,
                "beta": beta,
                "sigma": sigma,
                "config": config,
            }

        return q

    return oracle


def _test_blackbox_oracle():
    oracle = make_diffusion_oracle("nonlinear_high_noise", seed=7, noisy=False)
    out = oracle(s=1.0, T=2.0, return_full=True)

    assert np.isclose(out["q"], -3.2)
    assert np.isclose(out["q_true"], -3.2)
    assert np.isclose(out["q_s"], -3.6)
    assert np.isclose(out["q_T"], -2.0)
    assert np.isclose(out["noise"], 0.0)
    assert out["k_0"] == 1.0
    assert out["alpha"] == 0.5
    assert out["beta"] == 0.2
    assert out["sigma"] == 0.10

    noisy_oracle = make_diffusion_oracle("nonlinear_high_noise", seed=7, noisy=True)
    noisy_out = noisy_oracle(s=1.0, T=2.0, return_full=True)

    assert np.isclose(noisy_out["q_true"], -3.2)
    assert np.isclose(noisy_out["q_s"], -3.6)
    assert np.isclose(noisy_out["q_T"], -2.0)
    assert np.isclose(noisy_out["q"], noisy_out["q_true"] + noisy_out["noise"])

    array_oracle = make_diffusion_oracle("linear_no_noise", noisy=False)
    s_values = np.array([-1.0, 0.0, 1.0])
    T_values = np.array([0.5, 1.5, 2.5])
    array_out = array_oracle(s_values, T_values, return_full=True)

    expected_q = -1.5 * s_values
    expected_q_s = np.full_like(s_values, -1.5)
    expected_q_T = np.zeros_like(s_values)

    assert np.allclose(array_out["q"], expected_q)
    assert np.allclose(array_out["q_s"], expected_q_s)
    assert np.allclose(array_out["q_T"], expected_q_T)

    print("blackbox oracle tests passed")


if __name__ == "__main__":
    _test_blackbox_oracle()
