from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from itertools import product
from math import factorial
from pathlib import Path
from typing import Literal, Sequence

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.interpolate import griddata
from scipy.sparse import (
    bmat,
    block_diag,
    csc_matrix,
    coo_matrix,
    diags,
    hstack,
    kron,
)
from scipy.sparse.linalg import SuperLU, splu
from scipy.special import log_ndtr
from scipy.stats import norm

"""
INFO: 

Classically, GPs store and factorize a dense covariance matrix. That is, if there are N
latent variables, a direct Cholesky factorization costs O(N^3) operations, and O(N^2) in 
storage. This algo replaces the densely represented covariance matrix with an exact
sparse change of basis using the kernel packet, and then applies the monotonicity-constrained 
GP procedure to this sparse representation. 

In order, we:

    1. evaluate a 1D matern kernel and differentiate it as needed for deriv observations.
    2. construct a compactly supported set of basis functions (kernel packets) as local
       linear combinations of matern kernels. 
    3. tensorize 1D KP bases to a cartesian grid with several input dimensions
    4. build the joint prior over fxn vals and virtual derivs, then use EP on our sparse
       packet-basis linear systems
    5. load a complete (s, T) grid, standardize it, fit our model, reverse the 
       standardization, and report predictions and derivs. 

This script defines the GP kernel as: 
    
    k((s,T), (s*, T*)) := variance * matern_nu(s,s*; lengthscale)
                                   * matern_nu(T,T*; lengthscale)
    
    where instead of ARD lengthscales, one shared lengthscale is determined after s and T
    are normalized. This permits a separable product structure, thus enabling an nD KP
    factorization using Kronecker products.

We impose monotonicity constraints through virtual derivative-sign observations and EP: 

    P(m_j=1 | d_j) = Phi(sign * d_j / probit_scale)

For the flux convention we use in our original experiment, the script learns f=-q and uses
sign=+1 s.t. df/ds >= 0 <=> -df/ds <= 0, as necessary for physical correctness. Recall: the
EP procedure replaces each non-Gaussian probit factor with an unnormalized Gaussian site: 

    t_j(d_j) = exp(-0.5 tau_j d_j^2 + eta_j d_j),

and repeatedly moment-matches one site at a time. Due to the localized nature of the 
KP factorization, we can implement this procedure in parallel updates, followed by damping.

Recall: Kernel packets - 

For **ordered** 1D nodes x_i, a packet is

    phi_j(t) = sum_i A[i,j] k(t, x_i).

When nu = p + 1/2, the coefficients A[i,j] can be chosen so that interior packets have compact 
support. Each interior packet uses only 2p+3 = 2nu+2 nearby matern translates. Thus, coeff matrix 
A and the matrix of packet values are sparse (namely, banded) even though the original matern cov is 
dense.

For the joint fxn/deriv prior, we construct the sparse matrices B and C s.t.: 

    K_u C = B; K_u = B C^{-1}

Here, C is block diagonal in the packet coefficient matrices, and B is constructed from packet values 
and packet derivatives. Differentiation preserves a packet's support, so the derivative blocks remain 
sparse.

For Gaussian site precision T = diag(tau), the posterior mean satisfies
    
    (K_u^{-1} + T) mu = eta

Putting mu = B w and K_u = B C^{-1} yields the sparse system

    (C + T B) w = eta; mu = B w, 
    
which can be solved using efficient sparse solvers (LU, for example). 

The posterior covariance identity used for EP variances is

    Sigma = B (C + T B)^{-1}.

Notably, the code never needs to assemble ``K_u`` or ``K_u^{-1}`` as dense matrices, which is the 
secret sauce to its efficiency (hopefully :)). 
    
Important remarks: For this algo to work, 
    
    1. Kernel must be a separable product structure. 
    2. Matern parameter nu must be a half int, and at least 3/2.
    3. Each fxn-deriv axis must form a complete Cartesian grid. 
    4. Virtual derv observations must also form a Cartesian grid.
    5. For matern 5/2, every fxn and virtual axis needs at least 7 points.

"""

PacketKind = Literal["left", "interior", "right"]

"""
STEP 1: Construction of 1D matern kernel. 

"""

def half_integer_order(nu: float, *, atol:float = 1.0e-12) -> int:

    """
    Converts a half int matern smoothness (nu) to its polynomial order (p).

    Kernel packets exist for matern smoothness values in form nu = p + 1/2, where
    p = 0, 1, 2, ...

    For these vals, the 1D matern kernel is an exponential factor mult by a deg-p
    polynomial. The polynom form is what makes it possible to cancel the left
    and right exponential tails with finitely many local kernel translates.

    Parameters
    ----------
    nu : float
        matern smoothness param
    atol: float
        abs tol used to check floating point half-integrality

    Returns
    -------
    int
        p in nu = p + 1/2; i.e. polynomial order

    Raises
    ------
    ValueError
        if nu is NOT a nonneg half int
    """

    nu = float(nu)
    p = int(round(nu-0.5))
    if p < 0 or not np.isclose(nu, p+0.5, atol=atol, rtol=0.0):
        raise ValueError("Kernel packets require nu = p + 1/2; "
                         f"received nu = {nu}")
    return p

def _matern_polynomial_coefficients(p: int) -> np.ndarray:

    """
    Returns coefficients of half-int matern polynomial.

    Given nu = p + 1/2, and z = sqrt(2nu)|x-x*|/l, the unit
    variance matern correlation function becomes:

        k(x,x*) = exp(-z)P_p(z),

    where P_p is a deg-p polynomial. Coeff formula here
    is the closed form acquired from the 1/2 int bessel fxn.

    We store the array in increasing powers because numpy.polynomial
    expects [c_0, c_1, ... , c_p]
    """

    coefficients = np.empty(p+1, dtype=float)
    scale = factorial(p)/factorial(2 * p)
    for degree in range(p+1):
        coefficients[degree] = (
            scale
            * (2.0**degree)
            * factorial(2*p - degree)
            / (factorial(p - degree) * factorial(degree))
        )

    return coefficients

@dataclass(frozen=True) # instances immutable once created
class MaternHalfInteger1D:
    """
    unit var, 1D matern covariance

    class uses:

        k(x,x*) = exp(-z) P_p(z)
        z = lambda |x-x*|
        lambda = sqrt(2nu) / l
        nu = p + 1/2

    Rmk: global variance applied later to multidimensional product kernel
    to avoid multiplying global variance once per input dim.

    first and second derivs included because monotone GP treats derivs
    as latent Gaussian vars. for a differentiable GP, differentiating the
    cov gives the cross-cov values involving virtual deriv vars.

    """
    nu: float = 2.5
    lengthscale: float = 1.0

    def __post_init__(self) -> None:

        """
        validates two kernel hyperparams at construction time. immutable dataclass
        important here, since packet coeffs are derived from nu and lengthscale; packets
        would be inconsistent if either hyperparam is later modified.

        """

        half_integer_order(self.nu)
        if not np.isfinite(self.lengthscale) or self.lengthscale <= 0.0:
            raise ValueError("length scale must be finite and positive")

    @property
    def p(self) -> int:

        """
        polynomial order p given nu = p+1/2
        """

        return half_integer_order(self.nu)

    @property
    def packet_degree(self) -> int:

        """
        number of local matern translates in an interior packet.

        KP theory from chen, ding, and tuo gives minimial degree:

            2nu+2 = 2p+3

        for a matern-5/2 model, p=2, and degree is 7. this is why we
        require at least 7 KPs for a 5/2 kernel

        """
        return 2 * self.p + 3

    @property
    def decay_rate(self) -> float:

        """"
        return lambda = sqrt(2nu)/l in z = lambda|x-x*|
        
        packet constraints are written wrt exponential tails exp(+lambdax), 
        exp(-lambdax); hence, this decay rate lambda appears in both cov eval 
        and packet construction. 
        
        """

        return float(np.sqrt(2.0 * self.nu) / self.lengthscale)

    @property
    def coefficients(self) -> np.ndarray:
        """
        polynom coeffs for half int matern kernel
        """
        return _matern_polynomial_coefficients(self.p)

    def covariance(self, x: np.ndarray, x_prime: np.ndarray) -> np.ndarray:
        """
        return k(x, x*) for one-dimensional coordinate arrays. in particular,
        if x has n entries, x* has m entries, the resulting cov matrix is nxm
        with:

            cov[i,j] = exp(-z_ij)P_p(z_ij)
            z_ij = sqrt(2nu)|x_i - x_j*|/l

        cov has a unit marginal var, k(x,x) = 1. global signal variance only introduced
        when product-kernel blocks are assembled later.

        """

        x = np.asarray(x, dtype=float).reshape(-1)
        x_prime = np.asarray(x_prime, dtype=float).reshape(-1)

        # broadcasting creates every pairwise abs displacement at once
        z = self.decay_rate * np.abs(x[:, None] - x_prime[None, :])
        polynomial = np.polynomial.polynomial.polyval(z, self.coefficients)
        return np.exp(-z) * polynomial

    def covariance_derivative(self, x: np.ndarray, x_prime: np.ndarray) -> np.ndarray:
        """
        return partial k(x,x*) / partial x, pairwise

        putting k=exp(-z)P(z), z = lambda|x-x*|, the chain rule gives us:

            partial_x k = lambda sign(x-x*) exp(-z) [P'(z)-P(z)]

        this matrix provides cov(f'(x), f(x*)). by stationarity,

            cov(f(x), f'(x*)) = -cov(f'(x), f(x*))

        Rmk: matern-1/2 paths are not mean-square diff, so method requires nu>= 3/2
        for monotone deriv constraints.

        """

        if self.p < 1:
            raise ValueError("first derivatives require nu >= 3/2")
        x = np.asarray(x, dtype=float).reshape(-1)
        x_prime = np.asarray(x_prime, dtype=float).reshape(-1)
        delta = x[:, None] - x_prime[None, :]
        z = self.decay_rate * np.abs(delta)

        first_coefficients = np.polynomial.polynomial.polyder(self.coefficients)
        polynomial = np.polynomial.polynomial.polyval(z, self.coefficients)
        first = np.polynomial.polynomial.polyval(z, first_coefficients)
        derivative = (
                self.decay_rate
                * np.sign(delta)
                * np.exp(-z)
                * (first - polynomial)
        )
        derivative[delta == 0.0] = 0.0 # where x = x', no contribution to deriv (delta = 0)

        return derivative

    def covariance_second_derivative(self, x: np.ndarray, x_prime: np.ndarray) -> np.ndarray:
        """
        return partial^2 k(x,x*) / partial x^2, pairwise

        differentiating again:
            partial_x^2 k = lambda^2 exp(-z)[P''(z)-2P'(z)+P(z)]

        for a stationary covariance, (dep on x-x*)

            Cov(f'(x), f'(x*)) = (partial^2/partial_x partial_{x*}) k(x, x*)
                               = -(partial^2 k(x,x') / partial x^2) k(x, x*)

        """

        if self.p < 1:
            raise ValueError("second derivatives require nu >= 3/2")
        x = np.asarray(x, dtype=float).reshape(-1)
        x_prime = np.asarray(x_prime, dtype=float).reshape(-1)
        z = self.decay_rate * np.abs(x[:, None] - x_prime[None, :])

        first_coefficients = np.polynomial.polynomial.polyder(self.coefficients)
        second_coefficients = np.polynomial.polynomial.polyder(first_coefficients)
        polynomial = np.polynomial.polynomial.polyval(z, self.coefficients)
        first = np.polynomial.polynomial.polyval(z, first_coefficients)
        second = np.polynomial.polynomial.polyval(z, second_coefficients)
        return (
                self.decay_rate ** 2
                * np.exp(-z)
                * (second - 2.0 * first + polynomial)
        )

"""
STEP 2: exact 1D kernel packet basis.

"""
@dataclass(frozen=True)
class PacketColumn:
    """
    metadata for one column of the 1D packet basis

    a packet column represents

        phi_j(t) = sum_{i in I_j} coefficients[i] k(t, x_i).

    Info:

    - node_indices stores the local index set I_j
    - kind identifies whether the packet is compactly supported in the interior, or one-sided at
      a boundary
    - support bounds let prediction skip all packet columns that are known to be zero at a
      requested coordinate
    - constraint_residual records how accurately the computed coefficients cancel the matern tails
    """

    basis_index: int
    kind: PacketKind
    node_indices: np.ndarray
    coefficients: np.ndarray
    support_left: float
    support_right: float
    constraint_residual: float


class KernelPacketFactorization1D:
    """
    exact kernel-packet change of basis on one **ordered** axis

    Put K(t,x) := row vector of ordinary matern translates
    centered at the ordered nodes x_1,...,x_n. the packet basis is

        phi(t) = K(t,x) A, or phi_j(t) = sum_i A[i,j] k(t,x_i)

    For nu=p+1/2, each interior column of A has only 2p+3 nonzero
    coefficients. those coefficients cancel all exponential-polynomial tail
    terms to the left and right, making phi_j compactly supported.  The
    first and last p+1 columns are one-sided packets that represent the
    boundary behavior while keeping the basis square and invertible.

    at the training nodes, put Phi = phi(x), yielding

        Phi = K A, i.e. K = Phi A^{-1}.

    Both A and Phi are sparse and banded, even though K is dense
    !!! this is the basic computational engine used throughout !!!

    """
    def __init__(
        self,
        *,
        x: np.ndarray,
        kernel: MaternHalfInteger1D,
        coefficient_matrix: csc_matrix,
        packets: Sequence[PacketColumn],
        maximum_constraint_residual: float,
    ) -> None:

        """
        store a completed 1D packet construction

        this initializer packages the ordered nodes, the sparse coefficient matrix A,
        packet metadata, and the largest null-space residual.

        precomputed support endpoints are kept as sorted arrays so active packets can be
        found by binary search.

        """

        self.x = np.asarray(x, dtype=float).reshape(-1)
        self.kernel = kernel
        self.A = coefficient_matrix.tocsc()
        self.packets = tuple(packets)
        self.maximum_constraint_residual = float(maximum_constraint_residual)
        self.n = self.x.size
        self._left_bounds = np.asarray(
            [packet.support_left for packet in self.packets], dtype=float
        )
        self._right_bounds = np.asarray(
            [packet.support_right for packet in self.packets], dtype=float
        )

    @classmethod
    def build(
        cls,
        x: np.ndarray,
        *,
        nu: float,
        lengthscale: float,
    ) -> "KernelPacketFactorization1D":

        """
        constructs every packet column and assemble the sparse matrix A

        Algo
        ----
            1. confirm the axis is strictly ordered and that enough points exist
               for packets of degree 2nu+2 (need 7 per axis for nu=5/2)
            2. use _packet_layout to choose local node sets for left-sided,
               interior, and right-sided packets
            3. build the homogeneous tail-cancellation system for each node set; it has
               one fewer row than columns, so its null space should be 1D
            4. take normalized null vector from sol to homo system in 3 as the packet
               coefficients
            5. place those local coefficients into one sparse column of A

        on an equally spaced axis, all interior node configurations differ only
        by a translation. by chen, ding, and tuo, the packet constraints are translation
        invariant, so one interior coefficient vector is computed and reused.

        Returns
        -------
        KernelPacketFactorization1D
            the exact sparse coefficient matrix with packet support metadata

        """

        x = _validate_axis(x, "KP axis")
        kernel = MaternHalfInteger1D(nu=nu, lengthscale=lengthscale)
        p = kernel.p
        degree = kernel.packet_degree
        if x.size < degree:
            raise ValueError(
                f"nu={nu:g} requires at least {degree} points per KP axis; "
                f"received {x.size}."
            )

        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        packets: list[PacketColumn] = []
        maximum_residual = 0.0

        # translation invariance => 1 interior null-space solve is sufficient on an equally spaced axis
        equally_spaced = _is_equally_spaced(x)
        cached_interior: tuple[np.ndarray, float] | None = None

        for basis_index, kind, node_indices in _packet_layout(x.size, p):
            nodes = x[node_indices]
            if kind == "interior" and equally_spaced and cached_interior is not None:
                coefficients, residual = cached_interior
                coefficients = coefficients.copy()
            else:
                # the rows encode tail moments that must vanish. their 1D null vector furnishes
                # the local combo of matern translates forming the packet
                constraints = _packet_constraint_matrix(
                    nodes,
                    kernel=kernel,
                    kind=kind,
                )
                coefficients, residual = _null_vector(constraints)
                if kind == "interior" and equally_spaced:
                    cached_interior = (coefficients.copy(), residual)

            support_left, support_right = _packet_support(
                x,
                node_indices,
                kind,
            )
            packets.append(
                PacketColumn(
                    basis_index=basis_index,
                    kind=kind,
                    node_indices=node_indices.copy(),
                    coefficients=coefficients.copy(),
                    support_left=support_left,
                    support_right=support_right,
                    constraint_residual=residual,
                )
            )
            maximum_residual = max(maximum_residual, residual)

            rows.extend(node_indices.tolist())
            columns.extend([basis_index] * node_indices.size)
            values.extend(coefficients.tolist())

        # COO is convenient while collecting local entries; CSC is the
        # efficient format for sparse columns, Kronecker products, and LU
        A = coo_matrix(
            (values, (rows, columns)),
            shape=(x.size, x.size),
        ).tocsc()
        A.sum_duplicates()
        A.eliminate_zeros()

        if maximum_residual > 1.0e-8:
            warnings.warn(
                "a kernel-packet constraint system has residual "
                f"{maximum_residual:.3e}. rescale the axis or shorten the "
                "lengthscale if numerical errors are visible.",
                RuntimeWarning,
                stacklevel=2,
            )

        return cls(
            x=x,
            kernel=kernel,
            coefficient_matrix=A,
            packets=packets,
            maximum_constraint_residual=maximum_residual,
        )

    def packet_matrix(self, x_star: np.ndarray, derivative_order: int = 0) -> csc_matrix:
        """
        evaluate packet values or packet derivatives at target coordinates

        since differentiation is linear,

            phi_j^(r)(t) = sum_i A[i,j] partial_t^r k(t,x_i) for r=0,1,2

        thus, a derivative of a compactly supported packet has support contained in
        the same interval. hence, each target coordinate interacts with
        only a fixed number of packet columns, and this routine returns a sparse
        matrix rather than evaluating all n columns.

        Parameters
        ----------
        x_star:
            coordinates at which packet rows are required
        derivative_order:
            0 for packet values, 1 for first derivatives, and 2 for
            second derivatives with respect to the target coordinate

        Returns
        -------
        scipy.sparse.csc_matrix
            Matrix Phi_star with row r and column j equal to
            phi_j^(derivative_order)(x_star[r])
        """

        derivative_order = int(derivative_order)
        if derivative_order not in (0, 1, 2):
            raise ValueError("derivative_order must be 0, 1, or 2")

        x_star = np.asarray(x_star, dtype=float).reshape(-1)
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []

        for row, coordinate in enumerate(x_star):
            # compact support reduces a nominal O(n) row evaluation to a small
            # local stencil whose width depends on nu, not on n
            first, last = self._candidate_packet_range(float(coordinate))
            for basis_index in range(first, last + 1):
                packet = self.packets[basis_index]
                nodes = self.x[packet.node_indices]
                if derivative_order == 0:
                    kernel_values = self.kernel.covariance([coordinate], nodes)
                elif derivative_order == 1:
                    kernel_values = self.kernel.covariance_derivative(
                        [coordinate], nodes
                    )
                else:
                    kernel_values = self.kernel.covariance_second_derivative(
                        [coordinate], nodes
                    )
                value = (kernel_values @ packet.coefficients).item()
                if value != 0.0:
                    rows.append(row)
                    columns.append(basis_index)
                    values.append(value)

        matrix = coo_matrix(
            (values, (rows, columns)),
            shape=(x_star.size, self.n),
        ).tocsc()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        return matrix

    def _candidate_packet_range(self, coordinate: float) -> tuple[int, int]:
        """
        find the contiguous block of packets whose support contains a point

        packet supports are ordered along the axis.  two searchsorted calls
        thus locate the active block in logarithmic time without checking
        every packet.

        nearest-column fallback protects against roundoff at
        a support boundary and outside the finite node range.
        """

        # finite packet endpoints have exactly zero value, so strict support
        # inequalities are used
        first = int(np.searchsorted(self._right_bounds, coordinate, side="right"))
        last = int(np.searchsorted(self._left_bounds, coordinate, side="left") - 1)
        first = max(0, min(first, self.n - 1))
        last = max(0, min(last, self.n - 1))
        if last < first:
            nearest = min(max(int(np.searchsorted(self.x, coordinate)), 0), self.n - 1)
            return nearest, nearest
        return first, last


def _validate_axis(axis: np.ndarray, name: str) -> np.ndarray:

    """
    return a finite, strictly increasing one-dimensional node array.

    this is important because packet layout assumes neighboring indices are
    neighboring coordinates, and the support search assumes packet
    endpoints are sorted. duplicate nodes would also make the kernel basis
    linearly dependent.
    """

    axis = np.asarray(axis, dtype=float).reshape(-1)
    if axis.size == 0:
        raise ValueError(f"{name} is empty")
    if np.any(~np.isfinite(axis)):
        raise ValueError(f"{name} contains NaN or infinite values")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return axis


def _is_equally_spaced(x: np.ndarray) -> bool:

    """
    test if an axis is uniformly spaced to floating-point accuracy.

    on a uniform grid, every interior packet uses the same relative offsets.

    since packet coefficients are invariant to translating all nodes together,
    the expensive SVD for an interior packet can be cached and reused.
    """

    if x.size < 3:
        return True
    differences = np.diff(x)
    spacing = float((x[-1] - x[0]) / (x.size - 1))
    scale = max(1.0, abs(float(x[0])), abs(float(x[-1])), abs(spacing))
    tolerance = 256.0 * np.finfo(float).eps * scale
    return bool(np.max(np.abs(differences - spacing)) <= tolerance)


def _packet_layout(n: int, p: int):
    """
    render the local node stencil for every column in the packet basis.

    a square basis needs exactly n packet columns:

    1. p+1 left-sided packets near the left boundary
    2. n-2(p+1) compact interior packets, each using 2p+3 nodes
    3. p+1 right-sided packets near the right boundary

    boundary packets use progressively larger/smaller one-sided stencils. the
    layout follows the kernel-packet basis construction that captures boundary
    effects while retaining linear independence.
    """

    degree = 2 * p + 3

    # p+1 left-sided packets
    for basis_index in range(p + 1):
        node_count = p + 2 + basis_index
        yield basis_index, "left", np.arange(node_count, dtype=int)

    # interior compact packets
    for basis_index in range(p + 1, n - p - 1):
        start = basis_index - (p + 1)
        yield basis_index, "interior", np.arange(start, start + degree, dtype=int)

    # p+1 right-sided packets
    for offset, basis_index in enumerate(range(n - p - 1, n)):
        node_count = 2 * p + 2 - offset
        yield basis_index, "right", np.arange(n - node_count, n, dtype=int)


def _packet_support(
    x: np.ndarray,
    node_indices: np.ndarray,
    kind: PacketKind,
) -> tuple[float, float]:
    """
    return the theoretical support interval of a packet column.

    interior packets vanish outside their first and last local nodes. a left
    boundary packet extends to -infinity but vanishes to the right of its
    last node; a right boundary packet behaves symmetrically to the left.
    """

    if kind == "left":
        return -np.inf, float(x[node_indices[-1]])
    if kind == "right":
        return float(x[node_indices[0]]), np.inf
    return float(x[node_indices[0]]), float(x[node_indices[-1]])


def _packet_constraint_matrix(
    nodes: np.ndarray,
    *,
    kernel: MaternHalfInteger1D,
    kind: PacketKind,
) -> np.ndarray:
    """
    build the homogeneous equations that cancel a packet's tails

    for nu=p+1/2, each matern translate is a deg-p polynomial times
    an exponential on either side of its center. outside all selected nodes,
    a linear combination therefore has the form

        exp(-lambda*t) sum_{r=0}^p c_r t^r -- right tail
        exp(+lambda*t) sum_{r=0}^p d_r t^r -- left tail

    compact support is obtained by forcing every polynomial coefficient
    c_r and d_r to zero. these conditions can be expressed as weighted
    moment equations in the unknown translate coefficients. interior packets
    cancel both complete tails. one-sided packets cancel one complete tail and
    add just enough conditions on the other side to leave a one-dimensional
    null space.

    the nodes are shifted to a nearby center before exponentiation per chen, ding,
    and tuo. translation does not change the null space, but centering and row
    normalization greatly improve numerical conditioning.

    """

    nodes = np.asarray(nodes, dtype=float).reshape(-1)
    p = kernel.p

    if kind == "interior":
        center = 0.5 * (nodes[0] + nodes[-1])
    elif kind == "left":
        center = nodes[-1]
    else:
        center = nodes[0]
    z = kernel.decay_rate * (nodes - center)

    equations: list[np.ndarray] = []

    def add_rows(sign: int, maximum_degree: int) -> None:
        """
        append moment rows z^r exp(sign*z) for r=0,...,degree.

        subtracting the largest exponent scales a row family by a harmless
        nonzero constant and avoids overflow/underflow. a homogeneous system
        has the same null space after such row scaling.

        """

        if maximum_degree < 0:
            return
        exponent = sign * z
        exponential = np.exp(exponent - np.max(exponent))
        for degree in range(maximum_degree + 1):
            equations.append((z**degree) * exponential)

    # interior packets cancel all p+1 coefficients on each tail. the
    # one-sided formulas use the full constraints only on the vanishing side.
    if kind == "interior":
        add_rows(-1, p)
        add_rows(+1, p)
    elif kind == "right":
        add_rows(-1, p)
        add_rows(+1, nodes.size - p - 3)
    else:
        add_rows(+1, p)
        add_rows(-1, nodes.size - p - 3)

    matrix = np.asarray(equations, dtype=float)
    expected_shape = (nodes.size - 1, nodes.size)
    if matrix.shape != expected_shape:
        raise RuntimeError(
            f"internal packet-system shape error: {matrix.shape} != {expected_shape}"
        )
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0.0) or np.any(~np.isfinite(norms)):
        raise FloatingPointError("degenerate kernel-packet constraint equation")
    return matrix / norms[:, None]


def _null_vector(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """
    extract the unique packet coefficient vector from a null space.

    the constraint matrix has shape (m-1,m) and, theoretically, rank
    m-1. its null space is thus one dimensional. in an SVD

        M = U diag(s) V^T,

    the final row of V^T is the right singular vector associated with the
    zero singular value. its sign and scale are mathematically arbitrary, so
    this function normalizes the largest magnitude to one and chooses a
    deterministic sign. the returned relative residual is a diagnostic of
    finite-precision tail cancellation.

    """

    _, _, vh = svd(matrix, full_matrices=True, check_finite=False)
    coefficients = vh[-1].astype(float, copy=True)
    scale = float(np.max(np.abs(coefficients)))
    if scale == 0.0 or not np.isfinite(scale):
        raise FloatingPointError("failed to construct a nonzero packet")
    coefficients /= scale
    pivot = int(np.argmax(np.abs(coefficients)))
    if coefficients[pivot] < 0.0:
        coefficients *= -1.0
    residual = float(
        np.linalg.norm(matrix @ coefficients) / np.linalg.norm(coefficients)
    )
    return coefficients, residual

"""
STEP 3: Product KP basis construction on Cartesian grids.

"""

def _sparse_kronecker(factors: Sequence[csc_matrix]) -> csc_matrix:
    """
    form a sparse kron product in row-major convention; for a separable
    kernel and Cartesian grid:

        K = K_1 tensor K_2 tensor ... tensor K_d
        A = A_1 tensor A_2 tensor ... tensor A_d

    with C-order flattening, the final coordinate varies fastest, matching
    factor order used here. since every 1D packet matrix is sparse, the
    resulting kron product remains sparse for a fixed dimension d and packet
    degree 2p+3.

    """
    if not factors:
        raise ValueError("empty factors; at least one kron factor required")
    result = factors[0].tocsc()
    for factor in factors[1:]:
        result = kron(result, factor, format="csc")
    result.sum_duplicates()
    result.eliminate_zeros()
    return result

@dataclass(frozen=True)
class ProductKernelPacketBasis:
    """
    tensor-product packet basis for a separable multiD matern GP.

    if cov factors as:

        k(x,x*) = prod_r k_r(x_r, x_r*),

    and phi_{r, j_r} is a 1D packet basis on dimensional axis r, then:

        phi_j(x) = prod_r phi_{r, j_r}(x_r)

    is a multiD packet basis. on a cartesian grid, its coeff matrix is just
    the kron product A = tensor_r A_r. the script here uses one shared standardized
    lengthscale, where each coordinate is notably standardized separately to remove
    unit differences.

    """

    axes: tuple[np.ndarray, ...]
    factorizations: tuple[KernelPacketFactorization1D, ...]
    coefficient_matrix: csc_matrix
    shape: tuple[int, ...]
    size: int

    @classmethod
    def build(
        cls,
        axes: Sequence[np.ndarray],
        *,
        nu: float,
        lengthscale: float,
    ) -> "ProductKernelPacketBasis":
        """
        construct 1D packet bases and tensorize them. every axis validated and factorized
        independently. sparse coeff matrices are then combined using a kron product.

        resulting shape and size define the row-major mapping between multiple indices and
        one flattened latent-var index.

        """

        axes_tuple = tuple(
            _validate_axis(axis, f"axis[{index}]")
            for index, axis in enumerate(axes)
        )
        if not axes_tuple:
            raise ValueError("at least one axis is required")
        if half_integer_order(nu) < 1:
            raise ValueError("monotone derivative constraints require nu >= 3/2")

        factorizations = tuple(
            KernelPacketFactorization1D.build(
                axis,
                nu=nu,
                lengthscale=lengthscale,
            )
            for axis in axes_tuple
        )
        A = _sparse_kronecker(
            [factorization.A for factorization in factorizations]
        )
        shape = tuple(axis.size for axis in axes_tuple)
        return cls(
            axes=axes_tuple,
            factorizations=factorizations,
            coefficient_matrix=A,
            shape=shape,
            size=int(np.prod(shape)),
        )

    @property
    def dimension(self) -> int:
        """
        number of input coordinates in the product kernel
        """
        return len(self.axes)

    def matrix_on_grid(
            self,
            target_axes: Sequence[np.ndarray],
            derivative_orders: Sequence[int] | None = None,
    ) -> csc_matrix:
        """
        evaluate product packets on a complete cartesian target grid.

        a mixed partial deriv of a product packet factorizes coordinate by coordinate:

            D^alpha phi_j(x) = prod_r d^{alpha_r} phi_{r, j_r}(x_r)

        this yields the complete-grid evaluation matrix via the kron product of 1D
        packet/deriv matrices. this is much faster than constructing arbitrary
        point rows one by one.

        Rmk: derivative_orders is a multi_index such as (1,0) for d/ds, (0,1) for d/dT,
        or (1,1) for a mixed derivative

        """

        target_axes = tuple(
            _validate_axis(axis, f"axis[{index}]")
            for index, axis in enumerate(target_axes)
        )
        if len(target_axes) != self.dimension:
            raise ValueError("target grid has a different input dimension")
        orders = self._validate_orders(derivative_orders)
        factors = [
            factorization.packet_matrix(axis, derivative_order=order)
            for factorization, axis, order in zip(
                self.factorizations,
                target_axes,
                orders
            )
        ]
        return _sparse_kronecker(factors)

    def matrix(
        self,
        points: np.ndarray,
        derivative_orders: Sequence[int] | None = None,
    ) -> csc_matrix:
        """
        evaluate sparse product-packet rows at arbitrary paired points.

        unlike matrix_on_grid, rows here are arbitrary points (x_{i1}, ..., x_{id}),
        and do not form a cartesian product. for each row, matrix obtains the active
        1D packets on every axis, takes the cartesian product or those small active
        index sets, and multiplies their values. this is a sparse row-wise kron/khatri-rao
        construction used for prediction at unstructured locations.

        flat column indices use numpy.ravel_multi_index(..., order='C'), s.t. they are
        consistent with all training-grid vectors and kron matrices elsewhere in the file.

        """

        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1,-1)
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError(f"points must have shape (n, {self.dimension}); got {points.shape}")
        if np.any(~np.isfinite(points)):
            raise ValueError("points contain NaN or infinite values")
        orders = self._validate_orders(derivative_orders)

        axis_rows = [
            factorization.packet_matrix(
                points[:, axis_index],
                derivative_order=orders[axis_index],
            ).tocsr()
            for axis_index, factorization in enumerate(self.factorizations)
        ]

        output_rows: list[int] = []
        output_columns: list[int] = []
        output_values: list[float] = []

        for row in range(points.shape[0]):
            """
            each axis contributes only its locally supported packet columns.
            their cartesian product is still small for fixed dimension.
            
            """

            index_sets: list[np.ndarray] = []
            value_sets: list[np.ndarray] = []
            for axis_matrix in axis_rows:
                start = axis_matrix.indptr[row]
                stop = axis_matrix.indptr[row + 1]
                index_sets.append(axis_matrix.indices[start:stop])
                value_sets.append(axis_matrix.data[start:stop])

            if any(indices.size == 0 for indices in index_sets):
                continue

            for local_indices in product(
                    *[range(indices.size) for indices in index_sets]
            ):
                multi_index = tuple(
                    int(index_sets[axis][local_indices[axis]])
                    for axis in range(self.dimension)
                )
                flat_index = int(
                    np.ravel_multi_index(multi_index, self.shape, order="C")
                )
                value = float(
                    np.prod(
                        [
                            value_sets[axis][local_indices[axis]]
                            for axis in range(self.dimension)
                        ]
                    )
                )
                if value != 0.0:
                    output_rows.append(row)
                    output_columns.append(flat_index)
                    output_values.append(value)

        result = csc_matrix(
            (output_values, (output_rows, output_columns)),
            shape=(points.shape[0], self.size),
        )
        result.sum_duplicates()
        result.eliminate_zeros()
        return result

    def _validate_orders(
            self,
            derivative_orders: Sequence[int] | None,
    ) -> tuple[int, ...]:
        """
        normalize a derivative multi-index and reject unsupported orders.

        only orders zero, one, and two are needed for function values, first
        derivatives, derivative/derivative covariance, and mixed first
        derivatives in prediction.
        """

        if derivative_orders is None:
            return (0,) * self.dimension
        orders = tuple(int(value) for value in derivative_orders)
        if len(orders) != self.dimension:
            raise ValueError("one derivative order is required per input dimension")
        if any(order not in (0, 1, 2) for order in orders):
            raise ValueError("packet derivative orders must be 0, 1, or 2")
        return orders

"""
STEP 3: Monotone EP in the KP basis, at long last :)

"""
def _inverse_mills_ratio(z: np.ndarray) -> np.ndarray:
    """
    compute the inverse mills ratio phi(z)/Phi(z) stably

    the analytic EP moments for a probit factor contain this ratio.  directly dividing
    norm.pdf(z) by norm.cdf(z) is unreliable for large negative
    z because Phi(z) can underflow to zero.  working in log space,

        phi(z)/Phi(z) = exp(log phi(z) - log Phi(z)),

    avoids that failure; log_ndtr is SciPy's stable log-normal-CDF routine.
    """

    z = np.asarray(z, dtype=float)
    return np.exp(norm.logpdf(z) - log_ndtr(z))

@dataclass(frozen=True)
class TikhonovRegularization:
    """
    optional diagonal Gaussian precision added to the EP latent state

    the latent vector is u = [f(X); d(Z)].  this object adds

        exp(-0.5 * lambda * ||P u||^2)

    to the approximate posterior, where P selects function values,
    derivative values, or both. because the contribution is diagonal, it
    enters the same sparse transformed system as the observation and EP-site
    precisions; it does not destroy the kernel-packet sparsity.

    """

    enabled: bool = True
    strength: float = 1.0e-2
    target: Literal["joint", "fxn", "deriv"] = "deriv"

    def __post_init__(self) -> None:
        if not np.isfinite(self.strength) or self.strength < 0.0:
            raise ValueError("tikhonov strength must be finite and nonnegative")
        if self.target not in ("joint", "fxn", "deriv"):
            raise ValueError("tikhonov target must be 'joint', 'fxn', or 'deriv'")

    def precision_diag(self, n_function: int, n_derivative: int) -> np.ndarray:
        """
        return the diagonal precision contribution in latent ordering.
        n_function entries correspond to f(X)and the following n_derivative
        entries correspond to the virtual derivatives d(Z)

        """

        total = int(n_function) + int(n_derivative)
        result = np.zeros(total, dtype=float)
        if not self.enabled or self.strength == 0.0:
            return result
        if self.target == "joint":
            result[:] = self.strength
        elif self.target == "fxn":
            result[:n_function] = self.strength
        else:
            result[n_function:] = self.strength
        return result

@dataclass(frozen=True)
class MonotoneKPResult:
    """
    immutable quantities retained after EP has finished.

    posterior_packet_weights is the vector w from (C + TB) w = eta. this w here is the most
    useful predictive object because function and derivative posterior means are rendered by
    muliplying local packet rows by w. fxn/deriv means and deriv vars are also stored on
    respective training/virtual grids for diagnostics.

    recall: the final gaussian-site natural params describe EP's approx to the probit
    constraints.

    """

    posterior_packet_weights: np.ndarray
    posterior_mean_function: np.ndarray
    posterior_mean_derivative: np.ndarray
    posterior_variance_derivative: np.ndarray
    site_precision_derivative: np.ndarray
    site_eta_derivative: np.ndarray
    converged: bool
    iterations: int
    final_system_matrix: csc_matrix

class MonotoneKernelPacketGP:
    """
    monotone product-matern GP solved in a sparse packet basis.

    latent state: the model works with

        u = [f_X; d_Z], where d_Z = partial f(Z)/partial x_r,

    and where X is the cartesian observation grid and Z is a cartesian grid of virtual
    deriv locations. the prior is gaussian and includes all fxn/fxn, fxn/deriv, and deriv/deriv
    covariances obtained by differentiation of the produt matern kernel.

    likelihoods: real observations use

        y_i | f_i ~ N(f_i, noise_variance)

    each virtual derivative uses a soft sign likelihood

        Phi(monotone_sign * d_j / probit_scale)

    EP approximates the latter with Gaussian sites.  smaller probit_scale
    approaches a hard sign indicator but can make EP numerically more delicate.

    sparse algebra: two matrices are built once

        K_u C = B, and K_u = B C^{-1}

    C contains packet coefficients and B contains packet values and
    derivatives.  For any current EP site precision T, posterior moments
    are computed from C+TB rather than from a dense joint covariance.

    """

    def __init__(
            self,
            function_axes: Sequence[np.ndarray],
            *,
            derivative_axes: Sequence[np.ndarray] | None = None,
            nu: float = 2.5,
            lengthscale: float = 1.0,
            variance: float = 1.0,
            noise_variance: float = 1.0e-4,
            derivative_dim: int = 0,
            monotone_sign: int = 1,
            probit_scale: float = 1.0e-3,
            max_iter: int = 50,
            damping: float = 0.7,
            tolerance: float = 1.0e-5,
            ridge_precision: float = 1.0e-8,
            tikhonov: TikhonovRegularization | None = None,
            variance_batch_size: int = 32,
            verbose: bool = True,
    ) -> None:
        """
        configure grids, kernel hyperparameters, and EP controls.

        Parameters
        ----------
        function_axes:
            ordered coordinate arrays whose cartesian product contains the
            observed function values
        derivative_axes:
            ordered arrays defining the virtual derivative grid. defaults to
            the function axes but may be coarser or finer.
        nu, lengthscale, variance:
            product matern hyperparameters. lengthscale is shared across
            standardized dimensions; variance multiplies the full product
            once.
        noise_variance:
            observation variance in the same standardized output units as
            values passed to fit
        derivative_dim:
            coordinate whose derivative receives the sign constraint
        monotone_sign:
            +1 favors df/dx_r >= 0; -1 favors df/dx_r <= 0
        probit_scale:
            softness epsilon of Phi(sign*d/epsilon)
        max_iter, damping, tolerance:
            EP stopping and stabilization parameters. damping blends each new
            site with its old value to reduce oscillation
        ridge_precision:
            small diagonal Gaussian precision added to every latent variable.
            this is a numerical regularizer, not observation noise
        variance_batch_size:
            number of posterior covariance columns solved together when
            extracting derivative marginal variances
        verbose:
            print one convergence diagnostic per EP iteration

        construction immediately builds the two sparse matrices C and
        B because they depend only on grids and kernel hyperparameters, not
        on observed target values or current EP sites

        """

        self.function_basis = ProductKernelPacketBasis.build(
            function_axes,
            nu=nu,
            lengthscale=lengthscale,
        )
        if derivative_axes is None:
            derivative_axes = function_axes
        self.derivative_basis = ProductKernelPacketBasis.build(
            derivative_axes,
            nu=nu,
            lengthscale=lengthscale,
        )
        if self.function_basis.dimension != self.derivative_basis.dimension:
            raise ValueError("function and derivative grids have different dimensions")

        self.dimension = self.function_basis.dimension
        self.derivative_dim = int(derivative_dim)
        if not 0 <= self.derivative_dim < self.dimension:
            raise ValueError("derivative_dim is outside the input dimension")
        self.monotone_sign = int(monotone_sign)
        if self.monotone_sign not in (-1, 1):
            raise ValueError("monotone_sign must be +1 or -1")

        self.nu = float(nu)
        self.lengthscale = float(lengthscale)
        self.variance = float(variance)
        raw_noise_variance = np.asarray(noise_variance, dtype=float)
        if raw_noise_variance.size == 1:
            noise_vector = np.full(
                self.function_size,
                float(raw_noise_variance.reshape(-1)[0]),
                dtype=float,
            )
        elif raw_noise_variance.shape == self.function_shape:
            noise_vector = raw_noise_variance.reshape(-1, order="C").copy()
        else:
            noise_vector = raw_noise_variance.reshape(-1).copy()
            if noise_vector.size != self.function_size:
                raise ValueError("noise variance must be scalar, have function_shape, or "
                                 f"contain {self.function_size} values")
        self.noise_variance = noise_vector
        self.probit_scale = float(probit_scale)
        self.max_iter = int(max_iter)
        self.damping = float(damping)
        self.tolerance = float(tolerance)
        self.ridge_precision = float(ridge_precision)
        self.tikhonov = (
            TikhonovRegularization(enabled=False, strength=0.0)
            if tikhonov is None
            else tikhonov
        )
        self.variance_batch_size = int(variance_batch_size)
        self.verbose = bool(verbose)

        if not np.isfinite(self.variance) or self.variance <= 0.0:
            raise ValueError("variance must be finite and positive")
        if not np.all(np.isfinite(self.noise_variance)) or np.any(
            self.noise_variance <= 0.0
        ):
            raise ValueError("noise_variance must be finite and positive")
        if not np.isfinite(self.probit_scale) or self.probit_scale <= 0.0:
            raise ValueError("probit_scale must be finite and positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if not 0.0 < self.damping <= 1.0:
            raise ValueError("damping must lie in (0,1]")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if self.ridge_precision < 0.0:
            raise ValueError("ridge_precision must be nonnegative")
        if self.variance_batch_size <= 0:
            raise ValueError("variance_batch_size must be positive")

        self._C, self._B = self._build_joint_factorization()
        self.result_: MonotoneKPResult | None = None

    @property
    def function_shape(self) -> tuple[int, ...]:
        """
        cartesian shape expected for observed fxn-value array

        """
        return self.function_basis.shape

    @property
    def derivative_shape(self) -> tuple[int,...]:
        """
        cartesian shape for virtual deriv grid

        """
        return self.derivative_basis.shape

    @property
    def function_size(self) -> int:
        """
        number N of observed function latent variables (X+Z)

        """
        return self.function_basis.size

    @property
    def derivative_size(self) -> int:
        """
        number M of virtual deriv latent variables

        """
        return self.derivative_basis.size

    def _orders(self, order: int, dimension: int | None = None) -> tuple[int, ...]:
        """
        create a derivative multi-index with one nonzero coordinate

        for example, in two dimensions, _orders(1,0) returns (1,0) and _orders(2,1)
        returns (0,2). omitting dimension selects the coordinate upon which monotonicity is
        imposed.

        """
        orders = [0]*self.dimension
        axis = self.derivative_dim if dimension is None else int(dimension)
        orders[axis] = int(order)
        return tuple(orders)

    def _build_joint_factorization(self) -> tuple[csc_matrix, csc_matrix]:
        """
        build sparse C and B such that K_joint C = B

        put A_X as packet coeff matrix on observation grid X, and A_Z as packet
        coeff matrix on virtual grid Z. then,

            C = blockdiag(A_X, A_Z)

        if Phi_X(x) = K(x,X)A_X, and Phi_Z(x) = K(x,Z)A_Z, differentiation yields:

            K_ff A_X =  Phi_X(X)
            K_fd A_Z = -D_r Phi_Z(X)
            K_df A_X =  D_r Phi_X(Z)
            K_dd A_Z = -D_r^2 Phi_Z(Z)

        the minus signs comes in a stationary kernel depends on x-x*:
        differentiating the second argument is the negative of differentiating
        the first. stacking the blocks yields

            B = variance * [[Phi_X(X),   -D_r Phi_Z(X)],
                            [D_r Phi_X(Z), -D_r^2 Phi_Z(Z)]]

        packet derivatives retain local support, so both B and C are
        sparse even though the equivalent joint covariance K_joint is
        dense.

        """

        x_basis = self.function_basis
        z_basis = self.derivative_basis

        """
        each ensuing call is a sparse kron evaluation on a complete cartesian grid. 
        variable names describe row location, source packet grid, and deriv order.
        
        """

        phi_x_at_x = x_basis.matrix_on_grid(x_basis.axes)
        dphi_z_at_x = z_basis.matrix_on_grid(
            x_basis.axes,
            self._orders(1),
        )
        dphi_x_at_z = x_basis.matrix_on_grid(
            z_basis.axes,
            self._orders(1),
        )
        d2phi_z_at_z = z_basis.matrix_on_grid(
            z_basis.axes,
            self._orders(2),
        )

        """
        K_joint C = B. for a stationary kernel, d/dx* k(x,x*) = -d/dx k(x,x*)
        """
        B = self.variance * bmat(
            [
                [phi_x_at_x, -dphi_z_at_x],
                [dphi_x_at_z, -d2phi_z_at_z],
            ],
            format="csc",
        )
        C = block_diag(
            (
                x_basis.coefficient_matrix,
                z_basis.coefficient_matrix,
            ),
            format="csc",
        )
        B.sum_duplicates()
        B.eliminate_zeros()
        C.sum_duplicates()
        C.eliminate_zeros()
        return C,B

    @staticmethod
    def _factor_system(matrix: csc_matrix) -> SuperLU:
        """
        factor a sparse posterior system with fill-reducing column ordering.

        COLAMD approximately minimizes fill-in in sparse LU factors.  the
        matrix C+TB is not explicitly forced into a symmetric storage form,
        so a general sparse LU factorization is used rather than sparse
        Cholesky. the wrapped error message lists the most common causes of a
        numerically singular system.
        """

        try:
            return splu(matrix.tocsc(), permc_spec="COLAMD")
        except RuntimeError as error:
            raise np.linalg.LinAlgError(
                "The sparse KP posterior system is singular or unstable. "
                "Increase observation noise or ridge_precision, rescale the "
                "coordinates, or avoid nearly duplicate virtual points."
            ) from error

    def _posterior_moments(
            self,
            precision: np.ndarray,
            eta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, csc_matrix]:
        """
        compute Gaussian posterior means and derivative marginal variances

        precision is the diagonal of T and eta is the natural
        linear parameter in

            q(u) prop to N(u;0,K_u) exp(-0.5 u^T T u + eta^T u)

        putting K_u=B C^{-1} and mu=Bw gives

            Sw = eta, where S = C + TB, and
            mu=Bw

        the posterior covariance is

            Sigma = BS^{-1}

        EP needs only the M diagonal entries belonging to virtual
        derivatives, not the full (N+M)^2 matrix.  for derivative index
        j, the variance is row j of B dotted with column j of
        S^{-1}.

        columns are solved in batches to limit peak memory.

        Returns
        -------
        mean:
            posterior mean of [f_X; d_Z]
        derivative_variance:
            diagonal posterior variances for d_Z only
        weights:
            packet-space solution w used for prediction
        system:
            sparse matrix S=C+TB used in this update
        """

        total = self.function_size + self.derivative_size
        precision = np.asarray(precision, dtype=float).reshape(-1)
        eta = np.asarray(eta, dtype=float).reshape(-1)
        if precision.size != total or eta.size != total:
            raise ValueError("precision and eta have the wrong length")

        """
        left multiplication by diag(precision) scales rows of B, exactly
        forming the TB term in the transformed posterior equations
        """
        system = (self._C + diags(precision, format="csc") @ self._B).tocsc()
        system.sum_duplicates()
        system.eliminate_zeros()
        lu = self._factor_system(system)

        # first solve in packet coordinates, then map back to latent values
        weights = np.asarray(lu.solve(eta), dtype=float).reshape(-1)
        mean = np.asarray(self._B @ weights, dtype=float).reshape(-1)

        # Sigma = B (C + T B)^(-1).  EP only needs the derivative diagonal.
        n = self.function_size
        m = self.derivative_size
        derivative_variance = np.empty(m, dtype=float)
        derivative_rows = self._B[n:, :].tocsr()

        for start in range(0, m, self.variance_batch_size):
            """
            Solving S X = E for selected standard-basis columns gives the
            corresponding columns of S^{-1} without forming a dense inverse.
            """
            stop = min(start + self.variance_batch_size, m)
            width = stop - start
            rhs = np.zeros((total, width), dtype=float)
            rhs[n + np.arange(start, stop), np.arange(width)] = 1.0
            inverse_columns = np.asarray(lu.solve(rhs), dtype=float)
            local_block = np.asarray(
                derivative_rows[start:stop, :] @ inverse_columns,
                dtype=float,
            )
            derivative_variance[start:stop] = np.diag(local_block)

        derivative_variance = np.maximum(derivative_variance, 1.0e-14)
        return mean, derivative_variance, weights, system

    def fit(self, y: np.ndarray) -> "MonotoneKernelPacketGP":
        """
        fit the monotonicity-constrained GP by expectation propagation

        Gaussian observation sites:

        for y_i|f_i ~ N(f_i,sigma_n^2), natural parameters are:

            tau_i = 1/sigma_n^2
            eta_i = y_i/sigma_n^2

        these sites remain fixed throughout EP

        derivative sign sites:

        each probit factor Phi(sign*d/epsilon) is approximated by

            t(d) = exp(-0.5 tau_site d^2 + eta_site d)

        Given the current marginal q(d)=N(m,v), removing that site gives the
        cavity natural parameters

            tau_cav = 1/v - tau_site,
            eta_cav = m/v - eta_site.

        multiplying the cavity by the true probit factor gives a tilted
        distribution whose first two moments are analytic.  putting

            z = sign*m_cav/sqrt(v_cav+epsilon^2)
            lambda = phi(z)/Phi(z)

        those moments are

            m_hat = m_cav + sign*v_cav*lambda/sqrt(v_cav+epsilon^2)
            v_hat = v_cav - v_cav^2/(v_cav+epsilon^2) lambda(lambda+z)

        a new Gaussian site is chosen so cavity * site has these moments.
        all derivative sites are proposed from the same current posterior and
        then damped. iteration stops when the largest change in any site
        natural parameter falls below tolerance.

        Parameters
        ----------
        y:
            standardized function values, either with function_shape or as
            a C-order flattened vector.
        """

        y = np.asarray(y, dtype=float)
        if y.shape == self.function_shape:
            y_vector = y.reshape(-1, order="C")
        else:
            y_vector = y.reshape(-1)
            if y_vector.size != self.function_size:
                raise ValueError(
                    f"y must have shape {self.function_shape} or length "
                    f"{self.function_size}; got {y.shape}"
                )
        if np.any(~np.isfinite(y_vector)):
            raise ValueError("y contains NaN or infinite values")

        n = self.function_size
        m = self.derivative_size
        total = n + m
        observation_precision = 1.0 / self.noise_variance

        """
        fixed_* contains factors that do not change during EP:
        observation likelihoods plus the small ridge precision
        """
        fixed_precision = np.full(total, self.ridge_precision, dtype=float)
        fixed_precision[:n] += observation_precision
        fixed_eta = np.zeros(total, dtype=float)
        fixed_eta[:n] = observation_precision * y_vector

        """
        a zero natural parameter means the initial derivative site is the neutral factor 1, 
        so the first iteration starts from the unconstrained GP conditioned only on real 
        observations.
        
        """
        site_tau = np.zeros(m, dtype=float)
        site_eta = np.zeros(m, dtype=float)
        converged = False
        iterations = 0

        for iteration in range(1, self.max_iter + 1):
            iterations = iteration
            precision = fixed_precision.copy()
            precision[n:] += site_tau
            eta = fixed_eta.copy()
            eta[n:] += site_eta
            mean, derivative_variance, _, _ = self._posterior_moments(
                precision,
                eta,
            )

            """
            remove each site's contribution from its current one-dimensional
            Gaussian marginal to obtain the EP cavity distribution.
            
            """
            derivative_mean = mean[n:]
            tau_cavity = 1.0 / derivative_variance - site_tau
            eta_cavity = derivative_mean / derivative_variance - site_eta
            valid = tau_cavity > 1.0e-14

            proposed_tau = site_tau.copy()
            proposed_eta = site_eta.copy()
            if np.any(valid):
                variance_cavity = 1.0 / tau_cavity[valid]
                mean_cavity = eta_cavity[valid] / tau_cavity[valid]
                """
                the probit likelihood can be viewed as adding independent
                N(0, epsilon^2) threshold noise. this is why cavity variance
                and probit_scale^2 add inside the square root.
                """
                denominator = np.sqrt(
                    variance_cavity + self.probit_scale ** 2
                )
                z = self.monotone_sign * mean_cavity / denominator
                ratio = _inverse_mills_ratio(z)

                tilted_mean = mean_cavity + (
                        self.monotone_sign
                        * variance_cavity
                        / denominator
                        * ratio
                )
                tilted_variance = variance_cavity - (
                        variance_cavity ** 2
                        / (variance_cavity + self.probit_scale ** 2)
                        * ratio
                        * (ratio + z)
                )
                tilted_variance = np.maximum(tilted_variance, 1.0e-12)

                """
                convert the moment-matched tilted Gaussian back to the
                natural parameters of the site: site = tilted / cavity.
                """
                tau_new = 1.0 / tilted_variance - tau_cavity[valid]
                eta_new = tilted_mean / tilted_variance - eta_cavity[valid]

                # Negative site precision can occur from floating-point error.
                invalid_site = tau_new < 0.0
                tau_new[invalid_site] = 0.0
                eta_new[invalid_site] = 0.0
                proposed_tau[valid] = tau_new
                proposed_eta[valid] = eta_new

            """
            damping is a convex combination in natural-parameter space. It often prevents 
            oscillation when the sign constraint is sharp.
            """
            updated_tau = (
                    (1.0 - self.damping) * site_tau
                    + self.damping * proposed_tau
            )
            updated_eta = (
                    (1.0 - self.damping) * site_eta
                    + self.damping * proposed_eta
            )
            max_change = float(
                max(
                    np.max(np.abs(updated_tau - site_tau), initial=0.0),
                    np.max(np.abs(updated_eta - site_eta), initial=0.0),
                )
            )
            site_tau = updated_tau
            site_eta = updated_eta

            if self.verbose:
                violation_fraction = float(
                    np.mean(self.monotone_sign * derivative_mean < 0.0)
                )
                print(
                    f"EP iter {iteration:03d} | max site change "
                    f"{max_change:.3e} | mean-sign violations "
                    f"{violation_fraction:.2%}"
                )

            if max_change < self.tolerance:
                converged = True
                break

        """
        recompute once with the converged/latest sites so the stored posterior 
        is synchronized with the final site parameters.
        """
        precision = fixed_precision.copy()
        precision[n:] += site_tau
        eta = fixed_eta.copy()
        eta[n:] += site_eta
        mean, derivative_variance, weights, final_system = self._posterior_moments(
            precision,
            eta,
        )

        self.result_ = MonotoneKPResult(
            posterior_packet_weights=weights,
            posterior_mean_function=mean[:n].copy(),
            posterior_mean_derivative=mean[n:].copy(),
            posterior_variance_derivative=derivative_variance.copy(),
            site_precision_derivative=site_tau.copy(),
            site_eta_derivative=site_eta.copy(),
            converged=converged,
            iterations=iterations,
            final_system_matrix=final_system.copy(),
        )
        return self

    def _require_result(self) -> MonotoneKPResult:
        """
        return fitted state or raise a clear precondition error
        """

        if self.result_ is None:
            raise RuntimeError("fit must be called before prediction")
        return self.result_

    def _cross_packet_matrix(
            self,
            points: np.ndarray,
            prediction_derivative_dim: int | None,
    ) -> csc_matrix:
        """
        build the local packet row mapping posterior weights to predictions

        for a function prediction, the packet representation of

            Cov(f(x_*),u) C is [Phi_X(x_*), -D_r Phi_Z(x_*)]

        for partial_j f(x_*), differentiate this row with respect to
        prediction coordinate j:

            [D_j Phi_X(x_*), -D_j D_r Phi_Z(x_*)].

        if j=r the second block uses a second derivative; otherwise it uses
        a mixed derivative. multiplying the resulting sparse row by posterior
        packet weights w produces the desired posterior mean directly.
        """

        if prediction_derivative_dim is None:
            left_orders = (0,) * self.dimension
            right_orders = self._orders(1)
        else:
            prediction_derivative_dim = int(prediction_derivative_dim)
            if not 0 <= prediction_derivative_dim < self.dimension:
                raise ValueError("prediction derivative dimension is invalid")

            left = [0] * self.dimension
            left[prediction_derivative_dim] = 1
            left_orders = tuple(left)

            right = [0] * self.dimension
            if prediction_derivative_dim == self.derivative_dim:
                right[self.derivative_dim] = 2
            else:
                right[prediction_derivative_dim] = 1
                right[self.derivative_dim] = 1
            right_orders = tuple(right)

        left_matrix = self.function_basis.matrix(points, left_orders)
        right_matrix = self.derivative_basis.matrix(points, right_orders)
        return self.variance * hstack(
            [left_matrix, -right_matrix],
            format="csc",
        )

    def predict(self, points: np.ndarray) -> np.ndarray:
        """
        return the posterior mean E[f(points)|data,constraints]

        prediction is local in the packet basis: each point activates only a
        small number of compact packet columns, which are dotted with the
        already-computed packet weights. the returned values remain in the
        standardized output units used during fitting.
        """

        result = self._require_result()
        cross = self._cross_packet_matrix(points, None)
        return np.asarray(
            cross @ result.posterior_packet_weights,
            dtype=float,
        ).reshape(-1)

    def predict_derivative(self, points: np.ndarray, derivative_dim: int) -> np.ndarray:
        """
        return the posterior mean of a first partial derivative of f

        derivative_dim selects the prediction coordinate and need not equal
        the constrained coordinate. the experiment later applies the chain
        rule to convert derivatives from standardized coordinates back to raw
        physical units.

        """

        result = self._require_result()
        cross = self._cross_packet_matrix(points, int(derivative_dim))
        return np.asarray(
            cross @ result.posterior_packet_weights,
            dtype=float,
        ).reshape(-1)

    def virtual_monotonicity_probability(self) -> np.ndarray:
        """
        return q(sign*d_j > 0) for every virtual derivative

        under the EP Gaussian approximation d_j ~ N(m_j,v_j), the latent
        derivative sign probability is

            Phi(sign*m_j/sqrt(v_j)).

        this is a diagnostic of the fitted latent derivative itself; it does
        not include the additional probit_scale threshold noise used in the
        likelihood factor.
        """

        result = self._require_result()
        z = (
                self.monotone_sign
                * result.posterior_mean_derivative
                / np.sqrt(result.posterior_variance_derivative)
        )
        return norm.cdf(z)

    def sparsity_summary(self) -> dict[str, float | int]:
        """
        report sizes and nonzero counts of the reusable KP matrices.

        B_density is especially useful for checking that compact support is
        actually being exploited. it should become small as grid size grows
        while packet degree and input dimension remain fixed.
        """

        total = self.function_size + self.derivative_size
        return {
            "function_points": self.function_size,
            "derivative_points": self.derivative_size,
            "latent_size": total,
            "C_nnz": int(self._C.nnz),
            "B_nnz": int(self._B.nnz),
            "C_density": float(self._C.nnz / (total * total)),
            "B_density": float(self._B.nnz / (total * total)),
        }

"""
Appendix: Data and experiment helpers (standardization and organizing as cartesian grid)
          Flux provider is also located here

"""
@dataclass(frozen=True)
class Standardizer:
    """
    scalar affine standardization used for each axis

    to convert back derivatives, use the chain rule. for f_raw = y_mean + y_scale f_std,
    and z_s = (s - s_mean)/s_scale, we have:

        partial f_raw / partial s
          = (y_scale/s_scale) partial f_std / partial z_s

    """
    mean: float
    scale: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        """
        estimate population mean and standard deviation from an array

        the population convention ddof=0 is used because scaling is a
        deterministic preprocessing transformation, not an unbiased estimator
        of an unknown sampling variance.  A constant axis cannot be used in a
        multidimensional product kernel and is rejected.
        """

        values = np.asarray(values, dtype=float).reshape(-1)
        mean = float(np.mean(values))
        scale = float(np.std(values, ddof=0))
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            raise ValueError("cannot standardize a constant or invalid array")
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        """
        map raw values to zero-centered, unit-scale coordinates

        """

        return (np.asarray(values, dtype=float) - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """
        Undo transform and recover raw physical units

        """

        return self.mean + self.scale * np.asarray(values, dtype=float)

def _load_training_frame(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """
    load a defensive copy of the training table
    """

    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"training CSV does not exist: {path}")
    return pd.read_csv(path)


def _coerce_numeric_column(frame: pd.DataFrame, column: str) -> None:
    """
    convert one required column to finite floating-point values in place
    """

    if column not in frame.columns:
        raise ValueError(f"training CSV is missing required column {column!r}")
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"column {column!r} must be numeric") from error
    if np.any(~np.isfinite(values)):
        raise ValueError(f"column {column!r} contains NaN or infinite values")
    frame[column] = values


def _extract_cartesian_axes(
    frame: pd.DataFrame,
    s_column: str,
    temperature_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    verify that the CSV locations form one complete S x T grid.

    a multidimensional exact KP basis is a tensor product of one-dimensional
    bases. consequently, arbitrary scattered rows cannot be used directly:
    every pair (s_i,T_j) must appear exactly once. this function checks
    duplicates, expected row count, and actual grid occupancy.

    """

    if frame.duplicated(subset=[s_column, temperature_column]).any():
        raise ValueError(
            f"training CSV contains duplicate ({s_column},{temperature_column}) "
            "coordinates"
        )

    s_axis = np.unique(frame[s_column].to_numpy(dtype=float))
    temperature_axis = np.unique(frame[temperature_column].to_numpy(dtype=float))
    s_axis.sort()
    temperature_axis.sort()

    expected = int(s_axis.size * temperature_axis.size)
    if len(frame) != expected:
        raise ValueError(
            "exact two-dimensional KP requires a complete Cartesian grid: "
            f"found {len(frame)} rows but expected {expected} = "
            f"{s_axis.size} x {temperature_axis.size}"
        )

    occupancy = frame.assign(_kp_present=1).pivot(
        index=s_column,
        columns=temperature_column,
        values="_kp_present",
    )
    occupancy = occupancy.reindex(index=s_axis, columns=temperature_axis)
    if occupancy.isna().any().any():
        raise ValueError("not every Cartesian (s,T) coordinate pair is present")
    return s_axis, temperature_axis


def _grid_column(
    frame: pd.DataFrame,
    value_column: str,
    *,
    s_column: str,
    temperature_column: str,
    s_axis: np.ndarray,
    temperature_axis: np.ndarray,
) -> np.ndarray:
    """
    arrange a CSV column into the C-order tensor layout used by KP

    rows of the returned matrix correspond to s and columns correspond to
    T. flattening with order='C' therefore makes T vary fastest,
    matching the kron factor order used by ProductKernelPacketBasis
    """

    _coerce_numeric_column(frame, value_column)
    grid = frame.pivot(
        index=s_column,
        columns=temperature_column,
        values=value_column,
    ).reindex(index=s_axis, columns=temperature_axis)
    values = grid.to_numpy(dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError(
            f"column {value_column!r} does not provide one finite value at "
            "every Cartesian coordinate"
        )
    return values


class MonotoneGPKPFluxST:
    """
    physical-unit flux provider backed by a monotone kernel-packet GP

    the class owns all transformations between physical units and the
    standardized coordinates required for stable packet construction. the
    low-level GP models a latent function f; by default f=-q so a
    positive latent derivative encodes the physical condition dq/ds <= 0

    Parameters
    ----------
    training_csv:
        CSV path or pandas DataFrame containing a complete Cartesian grid
    s_column, temperature_column, q_column:
        column labels for the two coordinates and observed flux
    noise_column:
        optional per-row physical noise-standard-deviation column. set to
        None to ignore any CSV noise column.
    noise_std:
        optional scalar or per-row physical standard deviation. overrides
        noise_column. a per-row vector follows the original CSV row order;
        an array with shape (n_s,n_T) is already interpreted as grid order.
    learn_neg_flux:
        when true, fit f=-q and impose df/ds>=0. when false, fit
        f=q and impose df/ds<=0.  both choices represent dq/ds<=0
    nu:
        Half-integer matern smoothness. first-derivative constraints require
        nu>=3/2. the default is matern 5/2
    lengthscale:
        one shared lengthscale in standardized (s,T) coordinates. this is
        deliberately not ARD, so that we can use tensor prod structure
    variance:
        variance multiplying the full product kernel once
    n_virtual_per_axis:
        number of virtual sign-constraint locations along each coordinate
    probit_nu:
        softness of Phi(sign*d/probit_nu); smaller values approximate a
        harder derivative-sign constraint but can make EP less stable.
    ep_max_iter, ep_damping, ep_tol:
        EP controls
    jitter:
        tiny all-latent precision used only to stabilize sparse solves
    use_tikhonov, tikhonov_strength, tikhonov_target:
        optional target-selective quadratic regularization
    variance_batch_size:
        number of inverse columns solved together when EP extracts derivative
        marginal variances
    prediction_batch_size:
        Maximum number of query points evaluated in one sparse packet batch.
    verbose:
        Print EP iteration diagnostics.

    Rmk: for nu=5/2, each one-dimensional packet needs seven nodes. therefore,
    each training axis and each virtual axis must contain at least seven points.
    """

    def __init__(
        self,
        training_csv: str | Path | pd.DataFrame,
        *,
        s_column: str = "s",
        temperature_column: str = "T",
        q_column: str = "q_noisy",
        noise_column: str | None = "sigma",
        noise_std: float | np.ndarray | None = None,
        learn_neg_flux: bool = True,
        nu: float = 2.5,
        lengthscale: float = 1.0,
        variance: float = 1.0,
        n_virtual_per_axis: int = 10,
        probit_nu: float = 1.0e-3,
        ep_max_iter: int = 20,
        ep_damping: float = 0.7,
        ep_tol: float = 1.0e-5,
        jitter: float = 1.0e-10,
        use_tikhonov: bool = True,
        tikhonov_strength: float = 1.0e-2,
        tikhonov_target: Literal["joint", "fxn", "deriv"] = "deriv",
        variance_batch_size: int = 32,
        prediction_batch_size: int = 4096,
        verbose: bool = False,
    ) -> None:
        self.s_column = str(s_column)
        self.temperature_column = str(temperature_column)
        self.q_column = str(q_column)
        self.noise_column = None if noise_column is None else str(noise_column)
        self.learn_neg_flux = bool(learn_neg_flux)
        self.nu = float(nu)
        self.lengthscale = float(lengthscale)
        self.variance = float(variance)
        self.n_virtual_per_axis = int(n_virtual_per_axis)
        self.probit_nu = float(probit_nu)
        self.ep_max_iter = int(ep_max_iter)
        self.ep_damping = float(ep_damping)
        self.ep_tol = float(ep_tol)
        self.jitter = float(jitter)
        self.tikhonov = TikhonovRegularization(
            enabled=bool(use_tikhonov),
            strength=float(tikhonov_strength),
            target=tikhonov_target,
        )
        self.variance_batch_size = int(variance_batch_size)
        self.prediction_batch_size = int(prediction_batch_size)
        self.verbose = bool(verbose)

        self._validate_configuration()

        self.training_frame_: pd.DataFrame | None = None
        self.s_scaler_: Standardizer | None = None
        self.temperature_scaler_: Standardizer | None = None
        self.y_scaler_: Standardizer | None = None
        self.model_: MonotoneKernelPacketGP | None = None
        self.s_axis_raw_: np.ndarray | None = None
        self.temperature_axis_raw_: np.ndarray | None = None
        self.s_virtual_raw_: np.ndarray | None = None
        self.temperature_virtual_raw_: np.ndarray | None = None
        self.noise_variance_standardized_: np.ndarray | None = None

        self.fit(training_csv, noise_std=noise_std)

    def _validate_configuration(self) -> None:
        """
        reject inconsistent model controls before expensive construction

        """

        p = half_integer_order(self.nu)
        if p < 1:
            raise ValueError("first-derivative constraints require nu >= 3/2")
        if not np.isfinite(self.lengthscale) or self.lengthscale <= 0.0:
            raise ValueError("lengthscale must be finite and positive")
        if not np.isfinite(self.variance) or self.variance <= 0.0:
            raise ValueError("variance must be finite and positive")
        if self.n_virtual_per_axis < 1:
            raise ValueError("n_virtual_per_axis must be positive")
        if not np.isfinite(self.probit_nu) or self.probit_nu <= 0.0:
            raise ValueError("probit_nu must be finite and positive")
        if self.ep_max_iter < 1:
            raise ValueError("ep_max_iter must be at least one")
        if not 0.0 < self.ep_damping <= 1.0:
            raise ValueError("ep_damping must lie in (0,1]")
        if not np.isfinite(self.ep_tol) or self.ep_tol <= 0.0:
            raise ValueError("ep_tol must be finite and positive")
        if not np.isfinite(self.jitter) or self.jitter < 0.0:
            raise ValueError("jitter must be finite and nonnegative")
        if self.variance_batch_size < 1:
            raise ValueError("variance_batch_size must be positive")
        if self.prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be positive")

    def _noise_variance_grid(
        self,
        frame: pd.DataFrame,
        q_grid: np.ndarray,
        y_scale: float,
        noise_std: float | np.ndarray | None,
        s_axis: np.ndarray,
        temperature_axis: np.ndarray,
    ) -> np.ndarray:
        """
        convert physical observation standard deviations to GP variances.

        if no noise information is supplied, a small variance 1e-6 is used
        in standardized output units. otherwise, each physical standard
        deviation is divided by the output scale and squared. per-row noise
        therefore becomes an exact heteroscedastic diagonal Gaussian
        likelihood rather than being averaged into one scalar.

        """

        if noise_std is None and self.noise_column is not None:
            if self.noise_column in frame.columns:
                sigma_grid = _grid_column(
                    frame,
                    self.noise_column,
                    s_column=self.s_column,
                    temperature_column=self.temperature_column,
                    s_axis=s_axis,
                    temperature_axis=temperature_axis,
                )
            else:
                sigma_grid = None
        elif noise_std is None:
            sigma_grid = None
        else:
            sigma = np.asarray(noise_std, dtype=float)
            if sigma.size == 1:
                sigma_grid = np.full_like(q_grid, float(sigma.reshape(-1)[0]))
            elif sigma.shape == q_grid.shape:
                sigma_grid = sigma.copy()
            elif sigma.size == len(frame):
                temporary = frame.copy()
                temporary["_kp_explicit_noise"] = sigma.reshape(-1)
                sigma_grid = _grid_column(
                    temporary,
                    "_kp_explicit_noise",
                    s_column=self.s_column,
                    temperature_column=self.temperature_column,
                    s_axis=s_axis,
                    temperature_axis=temperature_axis,
                )
            else:
                raise ValueError(
                    "noise_std must be scalar, match the Cartesian grid shape, "
                    "or contain one value per CSV row"
                )

        if sigma_grid is None:
            return np.full(q_grid.shape, 1.0e-6, dtype=float)
        if np.any(~np.isfinite(sigma_grid)):
            raise ValueError("noise standard deviations contain NaN or infinity")
        if np.any(sigma_grid < 0.0):
            raise ValueError("noise standard deviations cannot be negative")

        standardized = (sigma_grid / float(y_scale)) ** 2
        return np.maximum(standardized, 1.0e-12)

    def fit(
        self,
        training_csv: str | Path | pd.DataFrame,
        *,
        noise_std: float | np.ndarray | None = None,
    ) -> "MonotoneGPKPFluxST":
        """
        fit the complete provider from one labeled Cartesian CSV table.

        the function performs all data ordering, unit standardization, virtual
        grid construction, sparse KP factorization, and EP fitting. No command
        line state or global variables are used, so multiple providers can be
        constructed independently in one process.
        """

        frame = _load_training_frame(training_csv)
        _coerce_numeric_column(frame, self.s_column)
        _coerce_numeric_column(frame, self.temperature_column)
        _coerce_numeric_column(frame, self.q_column)

        s_axis_raw, temperature_axis_raw = _extract_cartesian_axes(
            frame,
            self.s_column,
            self.temperature_column,
        )
        q_grid = _grid_column(
            frame,
            self.q_column,
            s_column=self.s_column,
            temperature_column=self.temperature_column,
            s_axis=s_axis_raw,
            temperature_axis=temperature_axis_raw,
        )

        p = half_integer_order(self.nu)
        minimum_axis_size = 2 * p + 3
        if s_axis_raw.size < minimum_axis_size or temperature_axis_raw.size < minimum_axis_size:
            raise ValueError(
                f"nu={self.nu:g} requires at least {minimum_axis_size} distinct "
                "training coordinates along both s and T"
            )
        if self.n_virtual_per_axis < minimum_axis_size:
            raise ValueError(
                f"nu={self.nu:g} requires n_virtual_per_axis >= "
                f"{minimum_axis_size}"
            )

        """
        the latent sign convention is the only difference between learning
        q and -q. all physical outputs are converted back in evaluate()
        
        """
        latent_grid_raw = -q_grid if self.learn_neg_flux else q_grid

        s_scaler = Standardizer.fit(s_axis_raw)
        temperature_scaler = Standardizer.fit(temperature_axis_raw)
        y_scaler = Standardizer.fit(latent_grid_raw)

        s_axis = s_scaler.transform(s_axis_raw)
        temperature_axis = temperature_scaler.transform(temperature_axis_raw)
        y_grid = y_scaler.transform(latent_grid_raw)

        noise_variance_grid = self._noise_variance_grid(
            frame,
            q_grid,
            y_scaler.scale,
            noise_std,
            s_axis_raw,
            temperature_axis_raw,
        )

        """
        virtual points encode only derivative signs; they are not artificial
        derivative magnitudes.  affine input scaling preserves their ordering.
        """
        s_virtual_raw = np.linspace(
            float(s_axis_raw[0]),
            float(s_axis_raw[-1]),
            self.n_virtual_per_axis,
        )
        temperature_virtual_raw = np.linspace(
            float(temperature_axis_raw[0]),
            float(temperature_axis_raw[-1]),
            self.n_virtual_per_axis,
        )
        s_virtual = s_scaler.transform(s_virtual_raw)
        temperature_virtual = temperature_scaler.transform(
            temperature_virtual_raw
        )

        # for f=-q, df/ds>=0. for f=q, df/ds<=0. both imply dq/ds<=0
        monotone_sign = +1 if self.learn_neg_flux else -1
        model = MonotoneKernelPacketGP(
            (s_axis, temperature_axis),
            derivative_axes=(s_virtual, temperature_virtual),
            nu=self.nu,
            lengthscale=self.lengthscale,
            variance=self.variance,
            noise_variance=noise_variance_grid,
            derivative_dim=0,
            monotone_sign=monotone_sign,
            probit_scale=self.probit_nu,
            max_iter=self.ep_max_iter,
            damping=self.ep_damping,
            tolerance=self.ep_tol,
            ridge_precision=self.jitter,
            tikhonov=self.tikhonov,
            variance_batch_size=self.variance_batch_size,
            verbose=self.verbose,
        ).fit(y_grid)

        self.training_frame_ = frame.copy()
        self.s_scaler_ = s_scaler
        self.temperature_scaler_ = temperature_scaler
        self.y_scaler_ = y_scaler
        self.model_ = model
        self.s_axis_raw_ = s_axis_raw.copy()
        self.temperature_axis_raw_ = temperature_axis_raw.copy()
        self.s_virtual_raw_ = s_virtual_raw
        self.temperature_virtual_raw_ = temperature_virtual_raw
        self.noise_variance_standardized_ = noise_variance_grid.copy()
        return self

    def _require_fitted(
        self,
    ) -> tuple[MonotoneKernelPacketGP, Standardizer, Standardizer, Standardizer]:
        """
        return fitted components or raise a clear provider-state error
        """

        if (
            self.model_ is None
            or self.s_scaler_ is None
            or self.temperature_scaler_ is None
            or self.y_scaler_ is None
        ):
            raise RuntimeError("MonotoneGPKPFluxST must be fit before evaluate()")
        return (
            self.model_,
            self.s_scaler_,
            self.temperature_scaler_,
            self.y_scaler_,
        )

    def evaluate(
        self,
        s_q: np.ndarray | float,
        T_q: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        evaluate q, dq/ds, and dq/dT in physical units

        s_q and T_q may be scalars or any broadcast-compatible arrays.
        Outputs have the broadcast shape; scalar inputs return arrays of shape
        (-1) to match typical flux-provider interfaces.
        """

        model, s_scaler, temperature_scaler, y_scaler = self._require_fitted()

        s = np.atleast_1d(np.asarray(s_q, dtype=float))
        temperature = np.atleast_1d(np.asarray(T_q, dtype=float))
        try:
            s, temperature = np.broadcast_arrays(s, temperature)
        except ValueError as error:
            raise ValueError("s_q and T_q must be broadcast-compatible") from error
        if np.any(~np.isfinite(s)) or np.any(~np.isfinite(temperature)):
            raise ValueError("query coordinates contain NaN or infinite values")

        output_shape = s.shape
        raw_points = np.column_stack([s.reshape(-1), temperature.reshape(-1)])
        standardized_points = np.column_stack(
            [
                s_scaler.transform(raw_points[:, 0]),
                temperature_scaler.transform(raw_points[:, 1]),
            ]
        )

        """
        query points are independent once EP packet weights are fixed. small
        batches limit temporary sparse-matrix memory for large quadrature sets.
        """
        count = standardized_points.shape[0]
        f_standardized = np.empty(count, dtype=float)
        df_dz_s = np.empty(count, dtype=float)
        df_dz_temperature = np.empty(count, dtype=float)
        for start in range(0, count, self.prediction_batch_size):
            stop = min(start + self.prediction_batch_size, count)
            batch = standardized_points[start:stop]
            f_standardized[start:stop] = model.predict(batch)
            df_dz_s[start:stop] = model.predict_derivative(batch, 0)
            df_dz_temperature[start:stop] = model.predict_derivative(batch, 1)

        """
        undo output scaling and apply the chain rule for each physical input:
            df_raw/ds = y_scale/s_scale * df_std/dz_s
        """
        f_raw = y_scaler.inverse_transform(f_standardized)
        df_ds_raw = (y_scaler.scale / s_scaler.scale) * df_dz_s
        df_dT_raw = (
            y_scaler.scale / temperature_scaler.scale
        ) * df_dz_temperature

        flux_sign = -1.0 if self.learn_neg_flux else 1.0
        q = flux_sign * f_raw
        dq_ds = flux_sign * df_ds_raw
        dq_dT = flux_sign * df_dT_raw
        return (
            q.reshape(output_shape),
            dq_ds.reshape(output_shape),
            dq_dT.reshape(output_shape),
        )

    def diagnostics(self) -> dict[str, float | int | bool | tuple[int, ...]]:
        """
        return convergence, monotonicity, and sparse-structure diagnostics

        """

        model, _, _, _ = self._require_fitted()
        result = model._require_result()
        probabilities = model.virtual_monotonicity_probability()
        summary = model.sparsity_summary()
        return {
            "ep_converged": bool(result.converged),
            "ep_iterations": int(result.iterations),
            "minimum_virtual_sign_probability": float(np.min(probabilities)),
            "mean_virtual_sign_probability": float(np.mean(probabilities)),
            "training_grid_shape": model.function_shape,
            "virtual_grid_shape": model.derivative_shape,
            **summary,
        }

__all__ = [
    "MonotoneGPKPFluxST",
    "MonotoneKernelPacketGP",
    "MaternHalfInteger1D",
    "KernelPacketFactorization1D",
    "ProductKernelPacketBasis",
    "TikhonovRegularization",
]

def _rmse(predicted: np.ndarray, expected: np.ndarray) -> float:
    """
    return root-mean-square error for equally shaped numeric arrays

    """
    predicted = np.asarray(predicted, dtype=float)
    expected = np.asarray(expected, dtype=float)
    return float(np.sqrt(np.mean((predicted - expected) ** 2)))


def _cartesian_training_frame(frame: pd.DataFrame, grid_size: int) -> pd.DataFrame:
    """
    return the input grid, or interpolate scattered observations to a grid
    """
    try:
        _extract_cartesian_axes(frame, "s", "T")
        return frame[["s", "T", "q_noisy", "sigma"]].copy()
    except ValueError as grid_error:
        if grid_size < 7:
            raise ValueError("grid_size must be at least 7 for Matern 5/2 packets")

        s_axis = np.linspace(float(frame["s"].min()), float(frame["s"].max()), grid_size)
        temperature_axis = np.linspace(
            float(frame["T"].min()), float(frame["T"].max()), grid_size
        )
        S, TT = np.meshgrid(s_axis, temperature_axis, indexing="ij")
        source_points = frame[["s", "T"]].to_numpy(dtype=float)

        interpolated: dict[str, np.ndarray] = {}
        for column in ("q_noisy", "sigma"):
            source_values = frame[column].to_numpy(dtype=float)
            values = griddata(source_points, source_values, (S, TT), method="linear")
            missing = ~np.isfinite(values)
            if np.any(missing):
                nearest = griddata(
                    source_points, source_values, (S, TT), method="nearest"
                )
                values[missing] = nearest[missing]
            interpolated[column] = values.ravel(order="C")

        warnings.warn(
            f"{grid_error}. Resampling q_noisy and sigma onto a "
            f"{grid_size} x {grid_size} Cartesian training grid.",
            stacklevel=2,
        )
        return pd.DataFrame(
            {
                "s": S.ravel(order="C"),
                "T": TT.ravel(order="C"),
                **interpolated,
            }
        )

def plot_predictions(
        s,
        T,
        q_pred,
        dq_ds_pred,
        dq_dT_pred,
        q_true,
        dq_ds_true,
        dq_dT_true,
):
    predicted = [q_pred, dq_ds_pred, dq_dT_pred]
    truth = [q_true, dq_ds_true, dq_dT_true]
    names = ["q", "dq/ds", "dq/dT"]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(15,12),
        constrained_layout=True,
    )

    for row, (name, true_values, predicted_values) in enumerate(
        zip(names, truth, predicted)
    ):
        error = predicted_values - true_values

        panels = [
            (true_values, f"True {name}"),
            (predicted_values, f"Predicted {name}"),
            (error, f"Error in {name}"),
        ]

        for column, (values, title) in enumerate(panels):
            contour = axes[row, column].tricontourf(
                s,
                T,
                values,
                levels=30,
                cmap="viridis" if column<2 else "coolwarm",
            )
            axes[row, column].set_title(title)
            axes[row, column].set_xlabel("s")
            axes[row, column].set_ylabel("T")
            fig.colorbar(contour, ax=axes[row, column])

    plt.show()

def main(csv_path: str | Path, grid_size: int = 15) -> None:
    """
    train from a labeled cartesian CSV and report errors against its truth
    columns. a_true and b_true are interpreted as dq/ds and dq/dT.

    """
    csv_path = Path(csv_path).expanduser().resolve()
    frame = _load_training_frame(csv_path)
    required_columns = (
        "s", "T", "q_true", "q_noisy", "a_true", "b_true", "sigma"
    )
    for column in required_columns:
        _coerce_numeric_column(frame, column)
    training_frame = _cartesian_training_frame(frame, grid_size)

    print(f"Training from: {csv_path}")
    print(f"Kernel-packet training grid: {grid_size} x {grid_size}")

    provider = MonotoneGPKPFluxST(
        training_frame,
        s_column="s",
        temperature_column="T",
        q_column="q_noisy",
        noise_column="sigma",
        learn_neg_flux=True,
        nu=2.5,
        lengthscale=1.0,
        variance=1.0,
        n_virtual_per_axis=15,
        # keep the demo quick. increase this for a production fit.
        ep_max_iter=50,
        ep_damping=0.7,
        ep_tol=1.0e-5,
        use_tikhonov=True,
        tikhonov_strength=1.0e-2,
        tikhonov_target="deriv",
        verbose=True,
    )

    """
    evaluate in the CSV's original row order. the provider internally restores
    physical units for both flux and derivatives.
    
    """
    s_query = frame["s"].to_numpy(dtype=float)
    temperature_query = frame["T"].to_numpy(dtype=float)
    q, dq_ds, dq_dT = provider.evaluate(s_query, temperature_query)
    plot_predictions(
        s_query,
        temperature_query,
        q,
        dq_ds,
        dq_dT,
        frame["q_true"].to_numpy(dtype=float),
        frame["a_true"].to_numpy(dtype=float),
        frame["b_true"].to_numpy(dtype=float),
    )

    print("\nErrors against CSV truth columns")
    print("--------------------------------")
    print(f"RMSE(q):     {_rmse(q, frame['q_true']):.6g}")
    print(f"RMSE(dq/ds): {_rmse(dq_ds, frame['a_true']):.6g}")
    print(f"RMSE(dq/dT): {_rmse(dq_dT, frame['b_true']):.6g}")
    print(f"fraction predicted dq/ds > 0: {np.mean(dq_ds > 0.0):.3%}")

    print("\nDiagnostics")
    print("-----------")
    for name, value in provider.diagnostics().items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    default_csv_path = Path(__file__).with_name("nonlinear_low_noise.csv")
    parser = argparse.ArgumentParser(
        description=(
            "Train the monotone kernel-packet GP from a complete Cartesian "
            "CSV with columns s,T,q_true,q_noisy,a_true,b_true,sigma."
        )
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=default_csv_path,
        help=(
            "path to the training and validation CSV "
            "(default: nonlinear_low_noise.csv beside this script)"
        ),
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=15,
        help="points per axis when resampling scattered data (default: 15)",
    )
    args = parser.parse_args()
    main(args.csv, grid_size=args.grid_size)
