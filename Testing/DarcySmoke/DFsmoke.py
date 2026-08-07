from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent   
sys.path.insert(0, str(PROJECT_ROOT))

from Methods.OracleDataMethods.Multidimensional.baseGP import GPFluxST
from Data.BlackBoxOracle.anisotropicDF import (
    AnisotropicDarcyForchheimerOracle,
    ergun_principal_parameters,
    rotation_2d,
    rotation_3d_xyz,
)

SEED = 42
DIM = 2
N_TRAIN = 100
N_TEST = 300

PRESSURE_GRADIENT_MAX = 1.0e5  # Pa/m
NOISE_LEVEL = "none"           # "none", "low", "medium", "high"


def exact_inverse_jacobian(oracle, grad_p, u):
    """
    Exact du/d(grad p) from the inverse-function theorem.
    """
    A = oracle.mu * np.linalg.inv(oracle.K)
    C = oracle.B23

    eta = np.sqrt(u @ C @ u)

    if eta < 1e-14:
        J_forward = A
    else:
        Cu = C @ u
        J_forward = (
            A
            + oracle.rho * eta * C
            + oracle.rho * np.outer(Cu, Cu) / eta
        )

    # F = -grad(p), so du/d(grad p) = -[dF/du]^{-1}
    return -np.linalg.inv(J_forward)


def main():
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------
    # build anisotropic Darcy-Forchheimer oracle
    # ------------------------------------------------------------

    if DIM == 2:
        hydraulic_lengths = [1.0e-3, 0.55e-3]
        R = rotation_2d(30.0)
    else:
        hydraulic_lengths = [1.0e-3, 0.70e-3, 0.45e-3]
        R = rotation_3d_xyz(
            alpha_deg=10.0,
            beta_deg=20.0,
            gamma_deg=30.0,
        )

    k, beta = ergun_principal_parameters(
        hydraulic_lengths_m=hydraulic_lengths,
        porosity=0.40,
    )

    oracle = AnisotropicDarcyForchheimerOracle(
        dim=DIM,
        k_principal=k,
        beta_principal=beta,
        rotation=R,
        noise_level=NOISE_LEVEL,
        seed=SEED,
    )

    # ------------------------------------------------------------
    # training data: grad(p) -> u
    # ------------------------------------------------------------

    grad_p_train = rng.uniform(
        -PRESSURE_GRADIENT_MAX,
        PRESSURE_GRADIENT_MAX,
        size=(N_TRAIN, DIM),
    )

    u_train = oracle.evaluate(grad_p_train)

    # Current GP requires both s and T.
    # For this first Darcy experiment, use
    #
    #       s := grad(p)
    #       T := 0
    #
    # because temperature is not part of the constitutive law.
    T_train = np.zeros((N_TRAIN, 1))

    gp = GPFluxST(
        s_train=grad_p_train,
        T_train=T_train,
        q_train=u_train,
        learn_neg_flux=False,
        n_restarts_optimizer=0,
    )

    # ------------------------------------------------------------
    # independent clean test set
    # ------------------------------------------------------------

    grad_p_test = rng.uniform(
        -PRESSURE_GRADIENT_MAX,
        PRESSURE_GRADIENT_MAX,
        size=(N_TEST, DIM),
    )

    # assess against the clean physical law.
    old_noise = oracle.noise_level
    old_fraction = oracle.noise_fraction

    oracle.noise_level = "none"
    oracle.noise_fraction = 0.0

    u_true = oracle.evaluate(grad_p_test)

    oracle.noise_level = old_noise
    oracle.noise_fraction = old_fraction

    T_test = np.zeros((N_TEST, 1))

    u_pred, du_dgradp_pred, _ = gp.evaluate(
        grad_p_test,
        T_test,
    )

    # ------------------------------------------------------------
    # velocity accuracy
    # ------------------------------------------------------------

    velocity_rmse = np.sqrt(
        np.mean((u_pred - u_true) ** 2)
    )

    relative_velocity_rmse = (
        np.linalg.norm(u_pred - u_true)
        / np.linalg.norm(u_true)
    )

    # ------------------------------------------------------------
    # exact inverse Jacobian accuracy
    # ------------------------------------------------------------

    J_true = np.stack([
        exact_inverse_jacobian(
            oracle,
            grad_p_test[i],
            u_true[i],
        )
        for i in range(N_TEST)
    ])

    jacobian_rmse = np.sqrt(
        np.mean((du_dgradp_pred - J_true) ** 2)
    )

    # ------------------------------------------------------------
    # physical constitutive residual
    # ------------------------------------------------------------

    residuals = np.stack([
        oracle.constitutive_residual(
            grad_p_test[i],
            u_pred[i],
        )
        for i in range(N_TEST)
    ])

    relative_constitutive_residual = (
        np.linalg.norm(residuals)
        / np.linalg.norm(grad_p_test)
    )

    print("\nDARCY-FORCHHEIMER GP SMOKE TEST")
    print("--------------------------------")
    print(f"dimension:                     {DIM}")
    print(f"noise level:                   {NOISE_LEVEL}")
    print(f"training points:               {N_TRAIN}")
    print(f"test points:                   {N_TEST}")
    print()
    print(f"velocity RMSE:                 {velocity_rmse:.6e}")
    print(f"relative velocity RMSE:        {relative_velocity_rmse:.6e}")
    print(f"Jacobian RMSE:                 {jacobian_rmse:.6e}")
    print(
        f"relative constitutive residual:{relative_constitutive_residual:.6e}"
    )


if __name__ == "__main__":
    main()