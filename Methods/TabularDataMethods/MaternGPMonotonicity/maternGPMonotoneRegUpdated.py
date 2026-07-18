"""
INFO

Matern GPR (nu = 5/2) with monotonicity info using low_noise_nonlinear
synthetic dataset.

Recall: our  nonlinear flux law is

    q_true = -(k_0*(1 + alpha*T**2) + beta*s**2)*s


Testing-interface provider:
    MonotoneGPFluxST(...).evaluate(s_q, T_q) returns
    (q, dq/ds, dq/dT) in physical units.

Implementation summary:
    1. load tabulated flux data CSV containing the following columns:
        s, T, x, k_0, alpha, beta, sigma, q_true, q_noisy, a_true, b_true
    2. train on
        input X = [s, T]
        output y = -q_noisy
    3. fit an ordinary sklearn Matern GP first to get reasonable hyperparams
        (lengthscale [l_s, l_T], variance [sigma_f^2])
    4. refit a custom GP incorporating virtual derivative observations
        stipulating d(-q)/(ds) >= 0
    5. compute and report validation errors for q, dq/ds, dq/dT using q_true,
        a_true, and b_true
            - In particular, compute RMSE for q, dq/ds, dq/dT

Important sign convention: hmmm....positive gradients?
    This script learns f = -q by default and imposes df/ds >= 0. If a dataset
    defines flux law without a minus sign, put learnNegFlux= False.

Remark: Derivatives are constrained at virtual derivative points;
        not a 100% guarantee every point in the (s,T) domain is
        monotone.

"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.special import log_ndtr
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
PRESETS: These can be changed for another dataset or experiment. 

"""

csvPath = "/Users/abeehamirza/Desktop/RIPS/nonlinear_low_noise.csv"
learnNegFlux = True
matern_nu = 2.5

# number of virtual derivative eval points in each input direction
nVirtualPerAxis = 5  # since we use 5 in a 2D (s,T) domain, 25 eval points; should try 100

# as nu --> 0, the probit function becomes more step-like
probit_nu = 1e-3  # less sharp than monotonicity paper (1e-6)

# expectation propagation presets; damping avoids oscillations in iterative procedure
EPMaxIter = 20  # would suggest 100
EPDamping = 0.7
EPTol = 1e-5

# numerical jitter for chol decomps
jitter = 1e-8

# random split seed
randomState = 42

# derivative_dim -> tells us which input coordinate wrt which we impose monotonicity
derivDim = 0  # 0 => s, so constrain df/ds >= 0

# optimizer for learning hyperparams
nRestartsOpti = 0  # should be 9 per what I've seen, but this is just for a test run

# optional tikhonov regularization for EP latent vector
useTikhonovRegularization = False
tikhonovStrength = 1e-2  # lambda term in L2 reg; try 1e-6 to 1e-2
tikhonovTarget = "deriv"  # "joint" reg f_X and d_Z; "fxn" reg f_X; "deriv" reg d_Z

"""
-----------------------------------------------------------------------------------
Matern 5/2 kernel with derivative covariance blocks; sklearn won't do this, so I had 
to myself :). 

Remarks:

    The Matern 5/2 kernel is:
        k(x, x') = (sigma_f^2) * (1 + sqrt(5)*r/l + 5(r^2)/3l^2)exp(-sqrt(5)*r/l))
          r = ||x-x'||
          l = length-scale parameter
          sigma_f^2 = signal variance

    Automatic Relevance Determination (ARD):

        Bayesian technique to identify and prune irrelevant features from 
        high-D datasets. Assigns a separate, learnable lengthscale param to 
        each input dimension. A large lengthscale means the function varies 
        slowly in that direction, suggesting that input is less important for 
        prediction.

        An ARD kernel uses one lengthscale per input dimension:
            i.e. rather than r^2 = [(s - s')^2 + (T - T')^2]/(l^2)
            ARD kernel uses one lengthscale per input dimension, like so
            r^2 = [(s - s')^2]/l_s^2 + [(T - T')^2]/l_T^2

-----------------------------------------------------------------------------------
"""


@dataclass
class Matern52:
    variance: float
    lengthscales: np.array

    def _pairwise_deltas_and_r(self, X: np.ndarray, X_prime: np.ndarray):
        """
        If X has N rows and X_prime has M rows, this function compares each
        x_i in X with each x_j' in X_prime.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            contains points x.
        X_prime : np.ndarray, shape (M, D)
            contains points x'.

        Returns
        -------
        delta: np.ndarray, shape (N, M, D)
            contains raw coordinate differences.
            delta[i,j,k] = X[i,k] - X_prime[j,k]

        scaled_delta: np.ndarray, shape (N,M,D)
            contains scaled coordinate differences.
            in particular, coordinate differences are divided by ARD lengthscales.
            scaledDelta[i,j,k] = (X[i,k] - X_prime[j,k])/lengthscales[k]

        r: np.ndarray, shape (N, M)
            contains scaled distance.
            r[i,j] = sqrt(sum_k scaled_delta[i, j, k]**2)

        """

        X = np.atleast_2d(np.asarray(X, dtype=float))
        X_prime = np.atleast_2d(np.asarray(X_prime, dtype=float))

        # delta[i.j,k] = X[i,k] - X_prime[j,k]
        delta = X[:, None, :] - X_prime[None, :, :]
        scaled_delta = delta / self.lengthscales
        r = np.sqrt(np.sum(scaled_delta ** 2, axis=2))

        return delta, scaled_delta, r

    def K(self, X: np.ndarray, X_prime: np.ndarray) -> np.ndarray:
        """
        Covariance between function values f(X) and f(X_prime).

        If X has N rows and X_prime has M rows, this fxn returns cov matrix
        between f(X) and f(X_prime).

        That is, returns a matrix whose [i,j] entry is:

            Cov(f(x_i), f(x_j')) = k(x_i, x_j')

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            contains points x.
        X_prime : np.ndarray, shape (M, D)
            contains points x'.

        Returns
        -------
        K : np.ndarray, shape (N, M)
            covariance matrix. K[i.j] =  Cov(f(x_i), f(x_j')) = k(x_i, x_j')

        """
        _, _, r = self._pairwise_deltas_and_r(X, X_prime)
        a = np.sqrt(5.0)

        return (self.variance * (1.0 + a * r + (a ** 2) * r ** 2 / 3.0) * np.exp(-a * r))

    def dK_dx(self, X: np.ndarray, X_prime: np.ndarray, dim: int) -> np.ndarray:
        """
        Covariance between df(X)/dX_dim, f(X_prime). Namely, provides
        derivative of k(x, x') wrt first argument x.

        That is, returns a matrix whose [i,j] entry is:

            Cov(df(x_i)/dx_dim, f(x_j')) = d/dx_dim k(x_i, x_j')

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            contains points x.
        X_prime : np.ndarray, shape (M, D)
            contains points x'.
        dim : int
            coordinate direction wrt which deriv is taken.
            for flux-law inputs X = [s,T], dim = 0 -> s, dim = 1 -> T

        Returns
        -------
        dK : np.ndarray, shape (N, M)
            cross-cov matrix between derivs at X and function vals at X_prime.
            dK[i,j] = d/dx_dim k(x_i, x_j')


        """

        delta, _, r = self._pairwise_deltas_and_r(X, X_prime)
        a = np.sqrt(5.0)

        """
        Let r be the scaled distance. 

        d/dx_dim k(x_i, x_j') = -variance * (a^2/3) * (1+a*r) * exp(-a*r)
                   * (x_dim - x_dim')/lengthscale_dim^2

        Notice we avoid division by r, so well-behaved expression when x = x'.
        """

        prefactor = -self.variance * (a ** 2 / 3.0) * (1.0 + a * r) * np.exp(-a * r)
        return prefactor * delta[:, :, dim] / (self.lengthscales[dim] ** 2)

    def dK_dx_prime(self, X: np.ndarray, X_prime: np.ndarray, dim: int) -> np.ndarray:
        """
        Covariance between f(X), df(X_prime)/dX_prime_dim). Namely, provides
        derivative of k(x, x') wrt second argument x'.

        For a stationary kernel, k depends on x - x'. Thus:

            d/dx'_dim k(x, x') = - d/dx_dim k(x, x')

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            contains points x.
        X_prime : np.ndarray, shape (M, D)
            contains points x'.
        dim : int
            coordinate direction wrt which deriv is taken.
            for flux-law inputs X = [s,T], dim = 0 -> s, dim = 1 -> T

        Returns
        -------
        dK : np.ndarray, shape (N, M)
            cross-cov matrix between function vals at X and derivs at X_prime.
            dK[i,j] = d/dx'_dim k(x_i, x_j')

        """
        return -self.dK_dx(X, X_prime, dim)

    def d2K_dxdx_prime(self, X: np.ndarray, X_prime: np.ndarray, dim1: int, dim2: int) -> np.ndarray:
        """
        Covariance between df(X)/dX_dim1, df(X_prime)/dX_prime_dim2).

        That is, returns a matrix whose [i,j] entry is:

            Cov(df(x_i)/dx_dim1, df(x_j')/dx_dim2) = [d^2/(dx_dim1 dx_dim 2)]k(x_i, x_j')

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            contains points x.
        X_prime : np.ndarray, shape (M, D)
            contains points x'.
        dim1 : int
            coordinate direction wrt which deriv of f(x) is taken.
            for flux-law inputs X = [s,T], dim = 0 -> s, dim = 1 -> T
        dim2 : int
            coordinate direction wrt which deriv of f(x') is taken.
            for flux-law inputs X = [s,T], dim = 0 -> s, dim = 1 -> T

        Returns
        -------
        d2K_dxdx_prime : np.ndarray, shape (N, M)
            cross-cov matrix between derivs at X and X_prime.
            d2K[i,j] = d2/(dx_dim1 dx'_dim2) (k(x_i, x_j'))

        """

        delta, _, r = self._pairwise_deltas_and_r(X, X_prime)
        a = np.sqrt(5.0)

        # need to specify separate lengthscales since dim1 need not = dim2
        l_x = self.lengthscales[dim1]
        l_xPrime = self.lengthscales[dim2]

        # define a kronecker delta to handle same-coord vs. cross-coord derivs
        same_dim = 1.0 if dim1 == dim2 else 0.0

        # Remark: if you have ?s abt 2nd derivative computation, ask Abeeha :)

        # same-coord deriv component, present iff dim1==dim2
        term_diag = (a ** 2 / 3.0) * (1.0 + a * r) * same_dim / (l_x ** 2)

        # outer prod cont; can be nonneg for same-coord & cross-coord deriv
        term_outer = (a ** 4 / 3.0) * delta[:, :, dim1] * delta[:, :, dim2] / (
                l_x ** 2 * l_xPrime ** 2)

        exp_term = np.exp(-a * r)

        return self.variance * exp_term * (term_diag - term_outer)


"""
UTILS
"""

@dataclass
class TikhonovRegularization:
    """

    Optional quadratic regularization for the EP latent vector.

    This adds the factor

        exp(-0.5 * strength * ||P u||^2)

    to the approximate posterior, where u = [f_X; d_Z], and P selects
    which component of u is regularized.

    Notably, PTP = I_{N+M} if both f_X and d_Z reg, I_{N} if only f_X reg,
                   I_{M} if only d_Z reg

    """

    enabled: bool = False
    strength: float = 0.0
    target: Literal["joint", "fxn", "deriv"] = "joint"

    def precision_diag(self, n_function: int, n_derivative: int) -> np.ndarray:

        """

        Returns the diagonal precision contribution lambda * diag(P^T P)

        """

        total = int(n_function) + int(n_derivative)
        diag = np.zeros(total, dtype=float)

        if (not self.enabled) or self.strength == 0.0:
            return diag

        if self.target == "joint":
            diag[:] = self.strength
        elif self.target == "fxn":
            diag[:n_function] = self.strength
        elif self.target == "deriv":
            diag[n_function:] = self.strength

        return diag


def make_virtual_grid(X_raw: np.ndarray, n_per_axis: int) -> np.ndarray:
    """
    Places virtual derivative points on a rectangular grid in the observed datarange.

    Note that these aren't real observations from the CSV, but locs at which we will
    ask the monotone GP to enforce the derivative sign condition df/ds >= 0.

    Parameters
    ----------
    X_raw : np.ndarray, shape (N, 2)
        raw, unstandardized input data. col 0 is s, col 1 is T
    n_per_axis : int
        number of grid points to place along each input direction

    Returns
    -------
    virtual grid: np.ndarray, shape (n_per_axis**2, 2)
        grid of virtual derivative locations in raw physical coordinates
        each row is a virtual point (s_virt, T_virt)

    """
    s_grid = np.linspace(X_raw[:, 0].min(), X_raw[:, 0].max(), n_per_axis)
    T_grid = np.linspace(X_raw[:, 1].min(), X_raw[:, 1].max(), n_per_axis)
    S, TT = np.meshgrid(s_grid, T_grid, indexing="xy")
    return np.column_stack([S.ravel(), TT.ravel()])


def rmse(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes RMSE.

    Parameters
    ----------
    y_hat : np.ndarray, shape (N, )
        predicted vals
    y_true : np.ndarray, shape (N, )
        true vals

    Returns
    -------
    RMSE : float

    """
    return float(np.sqrt(np.mean((np.asarray(y_hat) - np.asarray(y_true)) ** 2)))


def stable_inverse_mills(z: float) -> float:
    """
    Compute the inverse Mills ratio phi(z)/Phi(z) in a numerically stable manner.

    Here, phi(z) is the standard normal pdf, and Phi(z) is the standard normal CDF.

    The inv Mills ratio comes up in the EP moment-matching formulas for monotonicity
    likelihood Phi(d/probit_nu), where d is a latent deriv value such as df/ds at a
    virtual deriv point.

    A direct computation like norm.pdf(z) / norm.cdf(z) can be numerically unstable
    when z is very negative, because Phi(z) becomes extremely small and may underflow
    to zero. To avoid this, we compute exp(log(phi(z)) - log(Phi(z))) using scipy's
    stable log-CDF implementation, log_ndtr.

    Parameters
    ----------
    z : float
        standardized cavity mean used in EP update. in monotonicity calculations,
        we observe z = mean_cavity/sqrt(var_cavity + probit_nu**2)

    Returns
    -------
    ratio : float
        inverse Mills ratio

    """
    return float(np.exp(norm.logpdf(z) - log_ndtr(z)))


"""
-----------------------------------------------------------------------------------
EP Model  

Preliminaries:     

In this GP model, we consider the latent vector

    u = [f_X; d_Z] (N+M dimensional)

where f_X are our real training points (f(x_i), x_i = (s_i, T_i)) and d_Z are 
our virtual derivative points. Recall the ordinary GPR model where 

    y_i = f(x_i) + e, e is N(0, sigma_n^2)

The likelihood of our real training points is given by:  

    p(y_i | u_i) = N(y_i | u_i, sigma_n^2) 
                 ~ exp(-1/(2sigma_n^2)(u_i - y_i)^2)

    Expanding the term in the exponential, we get 

        -1/(2sigma_n^2)* u_i^2 + 1/sigma_n^2 * u_i * y_i - 1/(2sigma_n^2)y_i^2

    The final term in this expression won't play a role in our MLE procedure, 
    so we will consider 

        p(y_i | u_i) ~ exp(-0.5 * tau_i * u_i^2 + eta_i * u_i)

    where tau_i = 1/sigma_n^2, eta_i = y_i/sigma_n^2

The monotonicity likelihood is more complicated. For derivative entries of u,
we do not observe exact deriv values, but rather sign info. Namely, 

    d_j = df(z_j)/ds >= 0

Per the monotonicity paper, we will use a probit likeliood to represent this. 

    p(m_j | d_j) = Phi(d_j/probit_nu)

    If d_j is strongly positive, Phi(d_j/probit_nu) ~ 1
    If d_j is strongly negative, Phi(d_j/probit_nu) ~ 0

    As probit_nu -> 0, Phi(d_j/probit_nu) -> 1_{d_j >= 0} (step function)

The full posterior distribution for the latent vector u is thus given by:

    p(u | y, m) ~ p(u) * p(y | f_X) * p(m | d_Z)
                ~ N(0, K_joint) * prod(N(u_i, sigma_n^2)) * prod(Phi(u_N+j)/probit_nu)
                with i running from 1 to N, j running from 1 to M

    All but the monotonicity joint likelihood are Gaussian, making the full
    posterior analytically intractable. Fortunately, we can use EP to approximate
    prod(Phi(u_N+j)/probit_nu) using Gaussian sites.

The EP Approximation Algorithm:

    Replaces Phi(d_j/probit_nu) with a local Gaussian approx, called a site:

        t_j(d_j) ~ exp(-0.5 * tau_j * d_j^2 + eta_j * d_j)

    which is just an unnormalized Gaussian factor. 

    Putting this into our full posterior, we get an approximate posterior 
    q(u) ~ p(u):

        q(u) ~ N(0, K_joint) * prod(exp(-0.5 * tau_i * u_i^2 + eta_i * u_i))
        i now running from 1 to N+M

    Now, we can derive the approximate Gaussian posterior in full: 

        p(u) ~ exp(-0.5 * u^T K_joint^{-1} u)
        prod(exp(-0.5 * tau_i * u_i^2 + eta_i * u_i)) = exp(-0.5 * u^T diag(tau) u + eta^T u)

        therefore, 

            q(u) ~ exp(-0.5 * u^T [K^{-1} + diag(tau)] u + eta^T u)

            or q(u) ~ N(mu, Sigma), where mu = Sigma*eta, Sigma = [K^{-1} + diag(tau)]^{-1}

    The EP update for a single derivative site is formulated like so:

        Let d = u_{N+j}
        Let q(d) = N(mu_q, v_q) be current approx marginal

        The site approx is:

            t_j(d) ~ exp(-0.5 * tau_j * d^2 + eta_j * d)

        EP will temporarily remove this to form the cavity distribution:

            q_-j(d) ~ q(d)/t_j(d) ~ N(mu_cav, v_cav), 
            where tau_cav = 1/v_q - tau_j
                  eta_cav = mu_q/v_q - eta_j
            thus mu_cav = eta_cav/tau_cav; v_cav = 1/tau_cav

        Then, we construct the **tilted distribution** (yay:))

                Put the true monotonicity factor back in locally for the 
                derivative whose gaussian site we removed:

                    p_hat(d) ~ q_-j(d) * Phi(d/probit_nu)

                Although not gaussian, first 2 moments here can be computed 
                analytically. They are E(d), and Var(d). Then it chooses a 
                new Gaussian site matching these moments to replace the cavity
                site.

                To compute our moments, we will require Z = E(Phi(d/probit_nu))

                    Phi(d/probit_nu) = P(e <= d/probit_nu) e ~ N(0, 1)
                                     = P(probit_nu * e <= d)
                                     = P(d - probit_nu * e >= 0)

                    Since d ~ N(mu_cav, v_cav), and probit_nu * e ~ N(0, probit_nu^2):

                        d - probit_nu * e ~ N(mu_cav, v_cav + probit_nu^2)

                    Thus, Z = Phi(mu_cav / sqrt(v_cav + probit_nu^2)), 
                    or Z = Phi(z), z = mu_cav / sqrt(v_cav + probit_nu^2). 

                    Using this, we get the tilted mean and variance like so:

                        mu_hat = mu_cav + v_cav/ sqrt(v_cav + probit_nu^2) * lambda(z)
                        v_hat = v_cav - v_cav^2/(v_cav + probit_nu^2) * lambda(z)[z + lambda(z)]

                    If you have questions about how I derived this, please ask me :)
                    Lambda here is the inverse mills ratio phi(z)/Phi(z)

            Using the tilted distribution mean and variance, we turn it into a 
            new gaussian site N(mu_hat, v_hat), where our 
            new marginal = cavity * new site, so wrt our natural parameters:

                tau_new = 1/v_hat - tau_cav
                eta_new = mu_hat/v_hat - eta_cav

            So the EP update is choosing a Gaussian site which, when multiplied
            by the cavity, gives us a Gaussian matching the tilted mean and variance.

-----------------------------------------------------------------------------------
"""


@dataclass
class EPMonotoneGPResult:
    kernel: Matern52
    X_train: np.ndarray
    X_virtual: np.ndarray
    derivative_dim: int  # specifies the column idx of input var wrt which we enforce monotonicity
    K_joint: np.ndarray
    posterior_mean_joint: np.ndarray
    posterior_cov_joint: np.ndarray
    alpha_for_prediction: np.ndarray
    site_precision: np.ndarray
    site_eta: np.ndarray


class EPMonotoneGP:

    def __init__(
            self,
            kernel: Matern52,
            noise_variance: float,
            derivative_dim: int = 0,
            probit_nu: float = 1e-3,
            max_iter: int = 100,
            damping: float = 0.7,
            tol: float = 1e-5,
            jitter: float = 1e-8,
            tikhonov=TikhonovRegularization(enabled=useTikhonovRegularization,
                                            strength=tikhonovStrength,
                                            target=tikhonovTarget,
                                            ),
            verbose: bool = True,
    ):

        self.kernel = kernel
        self.noise_variance = float(noise_variance)
        self.derivative_dim = int(derivative_dim)
        self.probit_nu = float(probit_nu)
        self.max_iter = int(max_iter)
        self.damping = float(damping)
        self.tol = float(tol)
        self.jitter = float(jitter)
        self.tikhonov = tikhonov if tikhonov is not None else TikhonovRegularization()
        self.verbose = bool(verbose)

        self.result_: EPMonotoneGPResult | None = None

    def _joint_prior(self, X_train: np.ndarray, X_virtual: np.ndarray) -> np.ndarray:

        """
        Build prior covariance for u = [f_X = f_train, d_Z = d_virtual]

        """
        d = self.derivative_dim

        K_ff = self.kernel.K(X_train, X_train)

        # K_fd = Cov(f_X, d_Z)
        K_fd = self.kernel.dK_dx_prime(X_train, X_virtual, d)

        # K_df = Cov(d_Z, f_X)
        K_df = K_fd.T

        # K_dd = Cov(d_Z, d_Z)
        K_dd = self.kernel.d2K_dxdx_prime(X_virtual, X_virtual, d, d)

        K_joint = np.block([[K_ff, K_fd], [K_df, K_dd]])

        # this step symmetrizes K-joint given roundoff error
        # chol decomp expects symm matrix, so we prefix this and the jitter here
        K_joint = 0.5 * (K_joint + K_joint.T)
        K_joint += self.jitter * np.eye(K_joint.shape[0])

        return K_joint

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_virtual: np.ndarray):

        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float).ravel()
        X_virtual = np.asarray(X_virtual, dtype=float)

        N = X_train.shape[0]
        M = X_virtual.shape[0]
        L = N + M

        K_joint = self._joint_prior(X_train, X_virtual)

        """
        computes the prior precision K^{-1} using chol decomp.

        when L = N+M, this inverse-like solve is too expensive; 
        maybe use kernel packet

        """
        K_chol = cho_factor(K_joint, lower=True, check_finite=False)
        K_inv = cho_solve(K_chol, np.eye(L), check_finite=False)

        """
        site parameters are represented in natural form
             t_i(u_i) ∝ exp(-0.5*tau_i*u_i^2 + eta_i*u_i)
        for real observations, tau and eta are fixed gaussian likelihood terms
        for derivative sites, we start with 0 for both tau and eta; EP approx 
        will update these natural parameters.

        """
        tau = np.zeros(L)
        eta = np.zeros(L)

        tau[:N] = 1.0 / self.noise_variance
        eta[:N] = y_train / self.noise_variance

        tau[N:] = 0.0
        eta[N:] = 0.0

        """
        tikhonov reg enters as fixed gaussian factor exp(-0.5*strength*u^T*P^T*P*u); 
        separate from tau since tau stores observation likelihood and EP params
        """
        tikhonov_precision_diag = self.tikhonov.precision_diag(N, M)
        if self.verbose and np.any(tikhonov_precision_diag > 0.0):
            print(
                "Tikhonov regularization enabled: "
                f"target={self.tikhonov.target}, strength={self.tikhonov.strength:g}"
            )

        posterior_mean = np.zeros(L)
        posterior_cov = K_joint.copy()

        for ep_iter in range(self.max_iter):

            """
            current posterior: 
                q(u) ~ N(0, K_joint) * prod(exp(-0.5 * tau_i * u_i^2 + eta_i * u_i))
                i now running from 1 to N+M
            """
            precision = K_inv + np.diag(tau + tikhonov_precision_diag)
            precision = 0.5 * (precision + precision.T)  # symmetrize for chol!
            chol_precision = cho_factor(precision + self.jitter * np.eye(L),
                                        lower=True, check_finite=False)
            posterior_cov = cho_solve(chol_precision, np.eye(L), check_finite=False)
            posterior_mean = posterior_cov @ eta

            max_change = 0.0

            # update only derivative sites; observation sites stay fixed

            for j in range(M):

                idx = N + j

                # marginal q(u_idx) = N(mu_q, var_q)
                mu_q = posterior_mean[idx]
                var_q = max(float(posterior_cov[idx, idx]), 1e-14)

                # cav distribution q_-j; remove current gaussian site.
                tau_cav = 1.0 / var_q - tau[idx]
                eta_cav = mu_q / var_q - eta[idx]

                # tau is precision; must be greater than zero for calcs below
                if tau_cav <= 1e-14:
                    continue

                var_cav = 1.0 / tau_cav
                mu_cav = eta_cav / tau_cav

                # moment matching for tilted distribution: N(mu_cav, var_cav)*Phi(d/probit_nu)
                denom = np.sqrt(var_cav + self.probit_nu ** 2)
                z = mu_cav / denom
                ratio = stable_inverse_mills(z)

                mu_hat = mu_cav + (var_cav / denom) * ratio
                var_hat = var_cav - (var_cav ** 2 / (var_cav + self.probit_nu ** 2)) * (
                        ratio * (ratio + z)
                )
                var_hat = max(float(var_hat), 1e-12)

                # convert matched marginal back into a gaussian site.
                tau_new = 1.0 / var_hat - tau_cav
                eta_new = mu_hat / var_hat - eta_cav

                """
                EP may yield small negative site precision due to numerical error.
                the below corrects for this. note that if we clamp tau_new, we should
                also clamp eta_nu. otherwise, we get no quadratic push but some linear push
                from the eta_nu, which isn't a valid quadratic site.
                """
                tau_new = float(tau_new)
                eta_new = float(eta_new)

                if tau_new < 0.0:
                    tau_new = 0.0
                    eta_new = 0.0

                old_tau = tau[idx]
                old_eta = eta[idx]

                tau[idx] = (1.0 - self.damping) * tau[idx] + self.damping * tau_new
                eta[idx] = (1.0 - self.damping) * eta[idx] + self.damping * eta_new

                max_change = max(max_change, abs(tau[idx] - old_tau), abs(eta[idx] - old_eta))

            if self.verbose:
                print(f"EP iter {ep_iter + 1:02d}; max site change = {max_change:.3e}")

            """
            if EP site parameters aren't changing much anymore, terminate EP
            before maxiterations reached
            """
            if max_change < self.tol:
                break

        # posterior after EP convergence
        precision = K_inv + np.diag(tau + tikhonov_precision_diag)
        precision = 0.5 * (precision + precision.T)  # symmetrize for chol!
        chol_precision = cho_factor(precision + self.jitter * np.eye(L),
                                    lower=True, check_finite=False)
        posterior_cov = cho_solve(chol_precision, np.eye(L), check_finite=False)
        posterior_mean = posterior_cov @ eta

        """
        prediction mean can be written as K_*u K_uu^{-1} E_q[u].
        store alpha = K_uu^{-1} E_q[u], and compute prediction mean later
        """
        alpha_for_prediction = cho_solve(K_chol, posterior_mean, check_finite=False)

        self.result_ = EPMonotoneGPResult(
            kernel=self.kernel,
            X_train=X_train,
            X_virtual=X_virtual,
            derivative_dim=self.derivative_dim,
            K_joint=K_joint,
            posterior_mean_joint=posterior_mean,
            posterior_cov_joint=posterior_cov,
            alpha_for_prediction=alpha_for_prediction,
            site_precision=tau,
            site_eta=eta,
        )

        return self

    def predict_mean_and_gradient(
            self, X_star: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict the posterior mean of f and its gradient in model coordinates

        Parameters
        ----------
        X_star : np.ndarray, shape (N, D) or (D,)
            Query points in the same standardized coordinates used in fit

        Returns
        -------
        f_mean : np.ndarray, shape (N,)
            posterior mean of the standardized latent function
        gradient_mean : np.ndarray, shape (N, D)
            posterior mean of the derivatives with respect to each standardized
            input coordinate
        """
        if self.result_ is None:
            raise RuntimeError("EPMonotoneGP must be fit before prediction.")

        res = self.result_
        X_star = np.asarray(X_star, dtype=float)
        if X_star.ndim == 1:
            X_star = X_star.reshape(1, -1)
        if X_star.ndim != 2:
            raise ValueError("X_star must be a 1-D point or a 2-D array of points.")
        if X_star.shape[1] != res.X_train.shape[1]:
            raise ValueError(
                f"Expected {res.X_train.shape[1]} input columns; "
                f"received {X_star.shape[1]}."
            )
        if not np.all(np.isfinite(X_star)):
            raise ValueError("X_star contains NaN or infinite values.")

        constrained_dim = res.derivative_dim

        # Cov(f*, u), where u = [f(X_train), d_constrained(X_virtual)]
        K_star_f = res.kernel.K(X_star, res.X_train)
        K_star_d = res.kernel.dK_dx_prime(
            X_star, res.X_virtual, constrained_dim
        )
        K_star_u = np.hstack([K_star_f, K_star_d])
        f_mean = K_star_u @ res.alpha_for_prediction

        # Cov(df*/dx_dim, u) for every input coordinate
        gradient_mean = np.empty((X_star.shape[0], X_star.shape[1]), dtype=float)
        for dim in range(X_star.shape[1]):
            D_star_f = res.kernel.dK_dx(X_star, res.X_train, dim)
            D_star_d = res.kernel.d2K_dxdx_prime(
                X_star, res.X_virtual, dim, constrained_dim
            )
            D_star_u = np.hstack([D_star_f, D_star_d])
            gradient_mean[:, dim] = D_star_u @ res.alpha_for_prediction

        return f_mean, gradient_mean

    def predict_mean_and_derivative(
            self, X_star: np.ndarray, dim: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        predict f and one deriv in standardized model coordinates
        """
        f_mean, gradient_mean = self.predict_mean_and_gradient(X_star)
        dim = int(dim)
        if dim < 0 or dim >= gradient_mean.shape[1]:
            raise ValueError(
                f"dim must be between 0 and {gradient_mean.shape[1] - 1}; got {dim}."
            )
        return f_mean, gradient_mean[:, dim]

    def predict_mean_and_ds(self, X_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        backwards-compatible predictor for the constrained derivative
        """
        return self.predict_mean_and_derivative(X_star, self.derivative_dim)


class MonotoneGPFluxST:
    """
    physical-unit flux provider compatible with the shared testing interface

    The low-level :class:`EPMonotoneGP` works in standardized coordinates and
    models a latent function f.  This adapter owns the input/output scalers,
    optionally learns f = -q so the monotonicity constraint is positive, and
    exposes the common provider contract:

        q, dq_ds, dq_dT = provider.evaluate(s_q, T_q)

    all three returned arrays are in physical units
    """

    def __init__(
            self,
            s_train: np.ndarray,
            T_train: np.ndarray,
            q_train: np.ndarray,
            *,
            noise_std: float | np.ndarray | None = None,
            learn_neg_flux: bool = True,
            n_virtual_per_axis: int = 10,
            probit_nu: float = 1e-3,
            ep_max_iter: int = 20,
            ep_damping: float = 0.7,
            ep_tol: float = 1e-5,
            jitter: float = 1e-8,
            n_restarts_optimizer: int = 0,
            random_state: int = 42,
            use_tikhonov: bool = True,
            tikhonov_strength: float = 1e-2,
            tikhonov_target: Literal["joint", "fxn", "deriv"] = "deriv",
            verbose: bool = False,
    ):
        self.learn_neg_flux = bool(learn_neg_flux)
        self.n_virtual_per_axis = int(n_virtual_per_axis)
        self.probit_nu = float(probit_nu)
        self.ep_max_iter = int(ep_max_iter)
        self.ep_damping = float(ep_damping)
        self.ep_tol = float(ep_tol)
        self.jitter = float(jitter)
        self.n_restarts_optimizer = int(n_restarts_optimizer)
        self.random_state = int(random_state)
        self.tikhonov = TikhonovRegularization(
            enabled=bool(use_tikhonov),
            strength=float(tikhonov_strength),
            target=tikhonov_target,
        )
        self.verbose = bool(verbose)

        if self.n_virtual_per_axis < 1:
            raise ValueError("n_virtual_per_axis must be at least 1.")
        if self.ep_max_iter < 1:
            raise ValueError("ep_max_iter must be at least 1.")
        if self.probit_nu <= 0.0:
            raise ValueError("probit_nu must be positive.")
        if self.jitter <= 0.0:
            raise ValueError("jitter must be positive.")

        self.x_scaler_: StandardScaler | None = None
        self.y_scaler_: StandardScaler | None = None
        self.ordinary_gp_: GaussianProcessRegressor | None = None
        self.model_: EPMonotoneGP | None = None
        self.X_train_raw_: np.ndarray | None = None
        self.X_train_: np.ndarray | None = None
        self.X_virtual_raw_: np.ndarray | None = None
        self.X_virtual_: np.ndarray | None = None
        self.noise_variance_: float | None = None

        self.fit(s_train, T_train, q_train, noise_std=noise_std)

    @staticmethod
    def _training_arrays(
            s_train: np.ndarray,
            T_train: np.ndarray,
            q_train: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        s = np.asarray(s_train, dtype=float).reshape(-1)
        T = np.asarray(T_train, dtype=float).reshape(-1)
        q = np.asarray(q_train, dtype=float).reshape(-1)

        if not (s.size == T.size == q.size):
            raise ValueError(
                "s_train, T_train, and q_train must contain the same number "
                "of samples."
            )
        if s.size < 2:
            raise ValueError("At least two training samples are required.")
        if not (
                np.all(np.isfinite(s))
                and np.all(np.isfinite(T))
                and np.all(np.isfinite(q))
        ):
            raise ValueError("Training data contains NaN or infinite values.")

        return np.column_stack([s, T]), q

    @staticmethod
    def _standardized_noise_variance(
            noise_std: float | np.ndarray | None, y_scale: float
    ) -> float:
        # When no physical noise estimate is supplied, use a modest default in
        # standardized output units.  Supplying training_df["sigma"] is preferred.
        if noise_std is None:
            return 1e-6

        sigma = np.asarray(noise_std, dtype=float).reshape(-1)
        if sigma.size == 0 or not np.all(np.isfinite(sigma)):
            raise ValueError("noise_std must contain finite values.")
        if np.any(sigma < 0.0):
            raise ValueError("noise_std cannot contain negative values.")

        sigma_raw = float(np.mean(sigma))
        return max((sigma_raw / y_scale) ** 2, 1e-10)

    def fit(
            self,
            s_train: np.ndarray,
            T_train: np.ndarray,
            q_train: np.ndarray,
            *,
            noise_std: float | np.ndarray | None = None,
    ) -> "MonotoneGPFluxST":
        """
        fit the monotone GP from physical (s, T, q) training data
        """
        X_raw, q_raw = self._training_arrays(s_train, T_train, q_train)
        latent_raw = -q_raw if self.learn_neg_flux else q_raw

        x_scaler = StandardScaler().fit(X_raw)
        y_scaler = StandardScaler().fit(latent_raw.reshape(-1, 1))
        X_train = x_scaler.transform(X_raw)
        y_train = y_scaler.transform(latent_raw.reshape(-1, 1)).ravel()

        y_scale = float(y_scaler.scale_[0])
        if not np.isfinite(y_scale) or y_scale <= 0.0:
            raise ValueError("The output scale must be finite and positive.")
        noise_variance = self._standardized_noise_variance(noise_std, y_scale)

        sklearn_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(2),
            length_scale_bounds=(1e-3, 1e3),
            nu=2.5,
        )
        ordinary_gp = GaussianProcessRegressor(
            kernel=sklearn_kernel,
            alpha=noise_variance,
            normalize_y=False,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
        )
        ordinary_gp.fit(X_train, y_train)

        variance = float(ordinary_gp.kernel_.k1.constant_value)
        lengthscales = np.asarray(
            ordinary_gp.kernel_.k2.length_scale, dtype=float
        )
        kernel = Matern52(variance=variance, lengthscales=lengthscales)

        X_virtual_raw = make_virtual_grid(X_raw, self.n_virtual_per_axis)
        X_virtual = x_scaler.transform(X_virtual_raw)

        model = EPMonotoneGP(
            kernel=kernel,
            noise_variance=noise_variance,
            derivative_dim=0,
            probit_nu=self.probit_nu,
            max_iter=self.ep_max_iter,
            damping=self.ep_damping,
            tol=self.ep_tol,
            jitter=self.jitter,
            tikhonov=self.tikhonov,
            verbose=self.verbose,
        )
        model.fit(X_train, y_train, X_virtual)

        self.x_scaler_ = x_scaler
        self.y_scaler_ = y_scaler
        self.ordinary_gp_ = ordinary_gp
        self.model_ = model
        self.X_train_raw_ = X_raw
        self.X_train_ = X_train
        self.X_virtual_raw_ = X_virtual_raw
        self.X_virtual_ = X_virtual
        self.noise_variance_ = noise_variance

        if self.verbose:
            print("Ordinary GP kernel after hyperparameter optimization:")
            print(ordinary_gp.kernel_)

        return self

    def evaluate(
            self, s_q: np.ndarray, T_q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate q, dq/ds, and dq/dT at physical query points.

        s_q and T_q may be scalars or broadcast-compatible arrays.
        The outputs are NumPy arrays with the broadcast shape (and are always
        at least one-dimensional for scalar inputs).
        """
        if self.model_ is None or self.x_scaler_ is None or self.y_scaler_ is None:
            raise RuntimeError("MonotoneGPFluxST must be fit before evaluate().")

        s = np.atleast_1d(np.asarray(s_q, dtype=float))
        T = np.atleast_1d(np.asarray(T_q, dtype=float))
        try:
            s, T = np.broadcast_arrays(s, T)
        except ValueError as exc:
            raise ValueError("s_q and T_q must be broadcast-compatible.") from exc

        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(T))):
            raise ValueError("Query data contains NaN or infinite values.")

        output_shape = s.shape
        X_raw = np.column_stack([s.ravel(), T.ravel()])
        X_standardized = self.x_scaler_.transform(X_raw)

        f_standardized, gradient_standardized = (
            self.model_.predict_mean_and_gradient(X_standardized)
        )

        y_mean = float(self.y_scaler_.mean_[0])
        y_scale = float(self.y_scaler_.scale_[0])
        x_scale = np.asarray(self.x_scaler_.scale_, dtype=float)

        f_raw = y_mean + y_scale * f_standardized
        gradient_raw = (
                y_scale * gradient_standardized / x_scale.reshape(1, -1)
        )

        # f = -q when learn_neg_flux=True; otherwise f = q.
        flux_sign = -1.0 if self.learn_neg_flux else 1.0
        q = flux_sign * f_raw
        dq_ds = flux_sign * gradient_raw[:, 0]
        dq_dT = flux_sign * gradient_raw[:, 1]

        return (
            q.reshape(output_shape),
            dq_ds.reshape(output_shape),
            dq_dT.reshape(output_shape),
        )


"""
MAIN EXPERIMENT
"""


def main():
    csv_path = Path(csvPath).expanduser()

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = [
        "s",
        "T",
        "q_true",
        "q_noisy",
        "a_true",
        "b_true",
        "sigma",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if len(df) < 5:
        raise ValueError("The dataset must contain at least five rows.")

    # reject invalid numeric values before fitting
    if not np.all(np.isfinite(df[required_cols].to_numpy(dtype=float))):
        raise ValueError(
            "The required CSV columns contain NaN or infinite values."
        )

    # Split by row index so all columns remain aligned.
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=randomState,
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    """
    Fit the physical-unit provider.
    MonotoneGPFluxST handles:
        1. input standardization,
        2. output standardization,
        3. ordinary-GP hyperparameter fitting,
        4. virtual derivative-point construction,
        5. EP monotonicity fitting,
        6. conversion of predictions back to physical units.
    """
    provider = MonotoneGPFluxST(
        s_train=train_df["s"].to_numpy(dtype=float),
        T_train=train_df["T"].to_numpy(dtype=float),
        q_train=train_df["q_noisy"].to_numpy(dtype=float),
        noise_std=train_df["sigma"].to_numpy(dtype=float),
        learn_neg_flux=learnNegFlux,
        n_virtual_per_axis=nVirtualPerAxis,
        probit_nu=probit_nu,
        ep_max_iter=EPMaxIter,
        ep_damping=EPDamping,
        ep_tol=EPTol,
        jitter=jitter,
        n_restarts_optimizer=nRestartsOpti,
        random_state=randomState,
        use_tikhonov=useTikhonovRegularization,
        tikhonov_strength=tikhonovStrength,
        tikhonov_target=tikhonovTarget,
        verbose=True,
    )

    print("\nOrdinary sklearn GP kernel after hyperparameter optimization:")
    print(provider.ordinary_gp_.kernel_)

    s_test = test_df["s"].to_numpy(dtype=float)
    T_test = test_df["T"].to_numpy(dtype=float)

    q_test_true = test_df["q_true"].to_numpy(dtype=float)
    a_test_true = test_df["a_true"].to_numpy(dtype=float)  # dq/ds
    b_test_true = test_df["b_true"].to_numpy(dtype=float)  # dq/dT

    q_pred, dq_ds_pred, dq_dT_pred = provider.evaluate(
        s_test,
        T_test,
    )

    """
    accuracy against the known synthetic truth
    """
    print("\nValidation metrics on held-out test rows")
    print("------------------------------------------")
    print(f"RMSE(q):      {rmse(q_pred, q_test_true):.6g}")
    print(f"RMSE(dq/ds):  {rmse(dq_ds_pred, a_test_true):.6g}")
    print(f"RMSE(dq/dT):  {rmse(dq_dT_pred, b_test_true):.6g}")

    """
    monotonicity check
    
    when the model learns f = -q, the constrained positive derivative is
    df/ds = -dq/ds
    """
    if learnNegFlux:
        constrained_tangent_pred = -dq_ds_pred
        constrained_tangent_true = -a_test_true
        tangent_description = "-dq/ds"
    else:
        constrained_tangent_pred = dq_ds_pred
        constrained_tangent_true = a_test_true
        tangent_description = "dq/ds"

    predicted_violation_fraction = float(
        np.mean(constrained_tangent_pred < 0.0)
    )
    true_violation_fraction = float(
        np.mean(constrained_tangent_true < 0.0)
    )

    print("\nPhysical tangent check")
    print("----------------------")
    print(
        f"The constrained positive tangent is {tangent_description}."
    )
    print(
        "fraction predicted constrained tangent < 0: "
        f"{predicted_violation_fraction:.3%}"
    )
    print(
        "fraction true constrained tangent < 0: "
        f"{true_violation_fraction:.3%}"
    )


if __name__ == "__main__":
    main()

"""
Important note: 

MonotoneGPFluxST fits the monotone GP once and exposes the shared provider
interface

    q, dq_ds, dq_dT = provider.evaluate(s, T)

The inputs s and T are supplied in physical coordinates. The returned flux
and derivatives are also in physical units. The provider handles input
standardization, output standardization, chain-rule conversion, and the
f = -q sign convention internally.

This allows the Newton workflow to use one provider call for the flux and
both tangent components at each collection of quadrature points.

"""