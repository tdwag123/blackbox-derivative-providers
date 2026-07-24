import warnings
from itertools import product
from math import factorial
import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.sparse import bmat, block_diag, csc_matrix, coo_matrix, diags, hstack, kron
from scipy.sparse.linalg import splu
from scipy.special import log_ndtr

def half_integer_order(nu):
    nu = float(nu)
    p = int(round(nu - 0.5))
    if p < 0 or not np.isclose(nu, p + 0.5, atol=1.0e-12, rtol=0.0):
        raise ValueError("nu must be a nonnegative half integer")
    return p

def _matern_coefficients(p):
    coefficients = np.empty(p+1, dtype=float)
    scale = factorial(p)/factorial(2*p)
    for degree in range(p+1):
        coefficients[degree] = (
            scale
            * (2.0**degree)
            * factorial(2*p - degree)
            / (factorial(p-degree) * factorial(degree))
        )
    return coefficients

def _validate_axis(axis, name):
    axis = np.asarray(axis, dtype=float).reshape(-1)
    if axis.size == 0 or np.any(~np.isfinite(axis)):
        raise ValueError(f"{name} must contain finite values")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return axis

def _equally_spaced(axis):
    if axis.size < 3:
        return True
    spacing = (axis[-1] - axis[0])/(axis.size - 1)
    scale = max(1.0, abs(axis[0]), abs(axis[-1]), abs(spacing))
    tolerance = 256.0 * np.finfo(float).eps * scale
    return np.max(np.abs(np.diff(axis) - spacing)) <= tolerance

def _packet_layout(size, p):
    degree = 2*p+3
    for index in range(p+1):
        yield index, "left", np.arange(p+2+index)
    for index in range(p+1, size-p-1):
        start = index-p-1
        yield index, "interior", np.arange(start, start+degree)
    for offset, index in enumerate(range(size-p-1, size)):
        count = 2*p+2-offset
        yield index, "right", np.arange(size-count, size)

def _packet_constraints(nodes, kernel, kind):
    p = kernel.p
    if kind == "interior":
        center = 0.5*(nodes[0] + nodes[-1])
    elif kind == "left":
        center = nodes[-1]
    else: 
        center = nodes[0]
    
    scaled = kernel.decay_rate * (nodes-center)
    rows = []

    def add_rows(sign, maximum_degree):
        if maximum_degree < 0: 
            return
        exponent = sign*scaled
        exponential = np.exp(exponent - np.max(exponent))
        for degree in range(maximum_degree + 1):
            rows.append(scaled**degree * exponential)
    
    if kind == "interior":
        add_rows(-1, p)
        add_rows(1, p)
    elif kind == "right":
        add_rows(-1, p)
        add_rows(1, nodes.size - p - 3)
    else:
        add_rows(1, p)
        add_rows(-1, nodes.size - p - 3)
    
    matrix = np.asarray(rows, dtype=float)
    if matrix.shape != (nodes.size - 1, nodes.size):
        raise RuntimeError("invalid packet constraint system")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms==0.0) or np.any(~np.isfinite(norms)):
        raise FloatingPointError("invalid packet constraint system")
    return matrix/norms[:, None]

def _null_vector(matrix):
    _, _, vh = svd(matrix, full_matrices=True, check_finite=False)
    coefficients = vh[-1].astype(float, copy=True)
    scale = np.max(np.abs(coefficients))
    if not np.isfinite(scale) or scale == 0.0:
        raise FloatingPointError("packet construction failed")
    coefficients /= scale
    pivot = np.argmax(np.abs(coefficients))
    if coefficients[pivot] < 0.0:
        coefficients *= -1.0
    residual = np.linalg.norm(matrix @ coefficients) / np.linalg.norm(coefficients)
    return coefficients, float(residual)


def _sparse_kron(factors):
    if not factors:
        raise ValueError("at least one factor is required")
    result = factors[0].tocsc()
    for factor in factors[1:]:
        result = kron(result, factor, format="csc")
    result.sum_duplicates()
    result.eliminate_zeros()
    return result


def _inverse_mills_ratio(z):
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z)
    tail = z < -10.0

    if np.any(~tail):
        values = z[~tail]
        log_pdf = -0.5 * values**2 - 0.5 * np.log(2.0 * np.pi)
        result[~tail] = np.exp(log_pdf - log_ndtr(values))

    if np.any(tail):
        values = z[tail]
        inverse = -1.0 / values
        correction = (
            inverse
            - 2.0 * inverse**3
            + 10.0 * inverse**5
            - 74.0 * inverse**7
        )
        result[tail] = -values + correction

    return result


class MaternHalfInteger1D:
    def __init__(self, nu=2.5, lengthscale=1.0):
        self.nu = float(nu)
        self.lengthscale = float(lengthscale)
        self.p = half_integer_order(self.nu)
        if not np.isfinite(self.lengthscale) or self.lengthscale <= 0.0:
            raise ValueError("lengthscale must be positive")
        self.packet_degree = 2 * self.p + 3
        self.decay_rate = np.sqrt(2.0 * self.nu) / self.lengthscale
        self.coefficients = _matern_coefficients(self.p)
        self.first_coefficients = np.polynomial.polynomial.polyder(self.coefficients)
        self.second_coefficients = np.polynomial.polynomial.polyder(
            self.first_coefficients
        )

    def covariance(self, x, y):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        z = self.decay_rate * np.abs(x[:, None] - y[None, :])
        polynomial = np.polynomial.polynomial.polyval(z, self.coefficients)
        return np.exp(-z) * polynomial

    def covariance_derivative(self, x, y):
        if self.p < 1:
            raise ValueError("nu must be at least 3/2")
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        delta = x[:, None] - y[None, :]
        z = self.decay_rate * np.abs(delta)
        polynomial = np.polynomial.polynomial.polyval(z, self.coefficients)
        first = np.polynomial.polynomial.polyval(z, self.first_coefficients)
        result = (
            self.decay_rate
            * np.sign(delta)
            * np.exp(-z)
            * (first - polynomial)
        )
        result[delta == 0.0] = 0.0
        return result

    def covariance_second_derivative(self, x, y):
        if self.p < 1:
            raise ValueError("nu must be at least 3/2")
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        z = self.decay_rate * np.abs(x[:, None] - y[None, :])
        polynomial = np.polynomial.polynomial.polyval(z, self.coefficients)
        first = np.polynomial.polynomial.polyval(z, self.first_coefficients)
        second = np.polynomial.polynomial.polyval(z, self.second_coefficients)
        return (
            self.decay_rate**2
            * np.exp(-z)
            * (second - 2.0 * first + polynomial)
        )


class KernelPacketFactorization1D:
    def __init__(self, axis, nu=2.5, lengthscale=1.0):
        self.axis = _validate_axis(axis, "packet axis")
        self.kernel = MaternHalfInteger1D(nu, lengthscale)
        self.size = self.axis.size
        if self.size < self.kernel.packet_degree:
            raise ValueError("not enough points for the packet basis")

        rows = []
        columns = []
        values = []
        self.packet_nodes = []
        self.packet_coefficients = []
        left_bounds = []
        right_bounds = []
        maximum_residual = 0.0
        cached_interior = None
        equally_spaced = _equally_spaced(self.axis)

        for index, kind, node_indices in _packet_layout(
            self.size, self.kernel.p
        ):
            nodes = self.axis[node_indices]
            if kind == "interior" and equally_spaced and cached_interior is not None:
                coefficients, residual = cached_interior
                coefficients = coefficients.copy()
            else:
                coefficients, residual = _null_vector(
                    _packet_constraints(nodes, self.kernel, kind)
                )
                if kind == "interior" and equally_spaced:
                    cached_interior = coefficients.copy(), residual

            if kind == "left":
                left, right = -np.inf, self.axis[node_indices[-1]]
            elif kind == "right":
                left, right = self.axis[node_indices[0]], np.inf
            else:
                left, right = self.axis[node_indices[0]], self.axis[node_indices[-1]]

            self.packet_nodes.append(node_indices.copy())
            self.packet_coefficients.append(coefficients.copy())
            left_bounds.append(left)
            right_bounds.append(right)
            maximum_residual = max(maximum_residual, residual)
            rows.extend(node_indices.tolist())
            columns.extend([index] * node_indices.size)
            values.extend(coefficients.tolist())

        self.coefficient_matrix = coo_matrix(
            (values, (rows, columns)), shape=(self.size, self.size)
        ).tocsc()
        self.coefficient_matrix.sum_duplicates()
        self.coefficient_matrix.eliminate_zeros()
        self.left_bounds = np.asarray(left_bounds, dtype=float)
        self.right_bounds = np.asarray(right_bounds, dtype=float)

        if maximum_residual > 1.0e-8:
            warnings.warn("large kernel-packet residual", RuntimeWarning, stacklevel=2)

    def packet_matrix(self, points, derivative_order=0):
        derivative_order = int(derivative_order)
        if derivative_order not in (0, 1, 2):
            raise ValueError("derivative_order must be 0, 1, or 2")

        points = np.asarray(points, dtype=float).reshape(-1)
        rows = []
        columns = []
        values = []

        for row, point in enumerate(points):
            first = int(np.searchsorted(self.right_bounds, point, side="right"))
            last = int(np.searchsorted(self.left_bounds, point, side="left") - 1)
            first = max(0, min(first, self.size - 1))
            last = max(0, min(last, self.size - 1))
            if last < first:
                nearest = min(
                    max(int(np.searchsorted(self.axis, point)), 0), self.size - 1
                )
                first = nearest
                last = nearest

            for column in range(first, last + 1):
                node_indices = self.packet_nodes[column]
                nodes = self.axis[node_indices]
                if derivative_order == 0:
                    kernel_values = self.kernel.covariance([point], nodes)
                elif derivative_order == 1:
                    kernel_values = self.kernel.covariance_derivative([point], nodes)
                else:
                    kernel_values = self.kernel.covariance_second_derivative(
                        [point], nodes
                    )
                value = (kernel_values @ self.packet_coefficients[column]).item()
                if value != 0.0:
                    rows.append(row)
                    columns.append(column)
                    values.append(value)

        matrix = coo_matrix(
            (values, (rows, columns)), shape=(points.size, self.size)
        ).tocsc()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        return matrix

def _lengthscale_tuple(lengthscale, dimension):
    values = np.asarray(lengthscale, dtype=float).reshape(-1)
    if values.size == 1:
        values = np.full(dimension, float(values[0]))
    if values.size != dimension or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("invalid lengthscale")
    return tuple(float(value) for value in values)


class ProductKernelPacketBasis:
    def __init__(self, axes, nu=2.5, lengthscale=1.0):
        self.axes = tuple(
            _validate_axis(axis, f"axis {index}")
            for index, axis in enumerate(axes)
        )
        if not self.axes:
            raise ValueError("at least one axis is required")
        if half_integer_order(nu) < 1:
            raise ValueError("nu must be at least 3/2")
        self.dimension = len(self.axes)
        self.lengthscales = _lengthscale_tuple(lengthscale, self.dimension)
        self.factorizations = tuple(
            KernelPacketFactorization1D(axis, nu, axis_lengthscale)
            for axis, axis_lengthscale in zip(self.axes, self.lengthscales)
        )
        self.coefficient_matrix = _sparse_kron(
            [factor.coefficient_matrix for factor in self.factorizations]
        )
        self.shape = tuple(axis.size for axis in self.axes)
        self.size = int(np.prod(self.shape))

    def _orders(self, derivative_orders):
        if derivative_orders is None:
            return (0,) * self.dimension
        orders = tuple(int(order) for order in derivative_orders)
        if len(orders) != self.dimension or any(
            order not in (0, 1, 2) for order in orders
        ):
            raise ValueError("invalid derivative orders")
        return orders

    def matrix_on_grid(self, target_axes, derivative_orders=None):
        target_axes = tuple(
            _validate_axis(axis, f"target axis {index}")
            for index, axis in enumerate(target_axes)
        )
        if len(target_axes) != self.dimension:
            raise ValueError("target dimension mismatch")
        orders = self._orders(derivative_orders)
        return _sparse_kron(
            [
                factor.packet_matrix(axis, order)
                for factor, axis, order in zip(
                    self.factorizations, target_axes, orders
                )
            ]
        )

    def matrix(self, points, derivative_orders=None):
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError("invalid point array")
        if np.any(~np.isfinite(points)):
            raise ValueError("points must be finite")

        orders = self._orders(derivative_orders)
        axis_rows = [
            factor.packet_matrix(points[:, axis], orders[axis]).tocsr()
            for axis, factor in enumerate(self.factorizations)
        ]
        rows = []
        columns = []
        values = []

        for row in range(points.shape[0]):
            index_sets = []
            value_sets = []
            for matrix in axis_rows:
                start = matrix.indptr[row]
                stop = matrix.indptr[row + 1]
                index_sets.append(matrix.indices[start:stop])
                value_sets.append(matrix.data[start:stop])
            if any(indices.size == 0 for indices in index_sets):
                continue

            ranges = [range(indices.size) for indices in index_sets]
            for local_indices in product(*ranges):
                multi_index = tuple(
                    int(index_sets[axis][local_indices[axis]])
                    for axis in range(self.dimension)
                )
                column = int(
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
                    rows.append(row)
                    columns.append(column)
                    values.append(value)

        matrix = csc_matrix(
            (values, (rows, columns)), shape=(points.shape[0], self.size)
        )
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        return matrix


class MonotoneKernelPacketGP:
    def __init__(
        self,
        function_axes,
        derivative_axes=None,
        nu=2.5,
        lengthscale=1.0,
        variance=1.0,
        noise_variance=1.0e-4,
        observation_mask=None,
        derivative_dim=0,
        monotone_sign=1,
        probit_scale=1.0e-3,
        max_iter=50,
        damping=0.7,
        tolerance=1.0e-6,
        ridge_precision=1.0e-10,
        use_tikhonov=False,
        function_regularization=0.0,
        derivative_regularization=1.0e-2,
        variance_batch_size=32,
        verbose=False,
    ):
        self.function_basis = ProductKernelPacketBasis(
            function_axes, nu, lengthscale
        )
        if derivative_axes is None:
            derivative_axes = function_axes
        self.derivative_basis = ProductKernelPacketBasis(
            derivative_axes, nu, lengthscale
        )
        if self.function_basis.dimension != self.derivative_basis.dimension:
            raise ValueError("function and derivative dimensions differ")

        self.function_shape = self.function_basis.shape
        self.derivative_shape = self.derivative_basis.shape
        self.function_size = self.function_basis.size
        self.derivative_size = self.derivative_basis.size
        self.dimension = self.function_basis.dimension
        self.derivative_dim = int(derivative_dim)
        self.monotone_sign = int(monotone_sign)
        self.variance = float(variance)
        self.probit_scale = float(probit_scale)
        self.max_iter = int(max_iter)
        self.damping = float(damping)
        self.tolerance = float(tolerance)
        self.ridge_precision = float(ridge_precision)
        self.use_tikhonov = bool(use_tikhonov)
        self.function_regularization = float(function_regularization)
        self.derivative_regularization = float(derivative_regularization)
        self.variance_batch_size = int(variance_batch_size)
        self.verbose = bool(verbose)

        if not 0 <= self.derivative_dim < self.dimension:
            raise ValueError("invalid derivative dimension")
        if self.monotone_sign not in (-1, 1):
            raise ValueError("monotone_sign must be -1 or 1")
        if not np.isfinite(self.variance) or self.variance <= 0.0:
            raise ValueError("variance must be positive")
        if not np.isfinite(self.probit_scale) or self.probit_scale <= 0.0:
            raise ValueError("probit_scale must be positive")
        if self.max_iter < 1 or not 0.0 < self.damping <= 1.0:
            raise ValueError("invalid EP controls")
        if self.tolerance <= 0.0 or self.ridge_precision < 0.0:
            raise ValueError("invalid numerical controls")
        if self.variance_batch_size < 1:
            raise ValueError("variance_batch_size must be positive")
        if any(
            not np.isfinite(value) or value < 0.0
            for value in (
                self.function_regularization,
                self.derivative_regularization,
            )
        ):
            raise ValueError("regularization must be nonnegative")

        raw_noise = np.asarray(noise_variance, dtype=float)
        if raw_noise.size == 1:
            self.noise_variance = np.full(
                self.function_size, float(raw_noise.reshape(-1)[0])
            )
        elif raw_noise.shape == self.function_shape:
            self.noise_variance = raw_noise.reshape(-1, order="C").copy()
        else:
            self.noise_variance = raw_noise.reshape(-1).copy()
        if self.noise_variance.size != self.function_size:
            raise ValueError("noise size mismatch")
        if np.any(~np.isfinite(self.noise_variance)) or np.any(
            self.noise_variance <= 0.0
        ):
            raise ValueError("noise variance must be positive")
        
        if observation_mask is None:
            self.observation_mask = np.ones(self.function_size, dtype=bool)
        else: 
            mask = np.asarray(observation_mask, dtype=bool)
            if mask.shape == self.function_shape: 
                mask = mask.reshape(-1, order="C")
            else:
                mask = mask.reshape(-1)
            if mask.size != self.function_size or not np.any(mask):
                raise ValueError("invalid observation mask")
            self.observation_mask = mask.copy()
        
        derivative_kernel = self.derivative_basis.factorizations[
            self.derivative_dim
        ].kernel
        derivative_variance = -float(
            derivative_kernel.covariance_second_derivative([0.0], [0.0])[0, 0]
        )
        self.function_regularization_precision = 0.0
        self.derivative_regularization_precision = 0.0
        if self.use_tikhonov:
            self.function_regularization_precision = (
                self.function_regularization / self.variance
            )
            self.derivative_regularization_precision = (
                self.derivative_regularization
                / (self.variance * derivative_variance)
            )

        self.C, self.B = self._joint_factorization()
        self.posterior_packet_weights = None
        self.posterior_mean_function = None
        self.posterior_mean_derivative = None
        self.posterior_variance_derivative = None
        self.site_precision_derivative = None
        self.site_eta_derivative = None
        self.converged = False
        self.iterations = 0
        self.final_site_change = np.inf

    def _derivative_orders(self, order, dimension=None):
        orders = [0] * self.dimension
        axis = self.derivative_dim if dimension is None else int(dimension)
        orders[axis] = int(order)
        return tuple(orders)

    def _joint_factorization(self):
        x_basis = self.function_basis
        z_basis = self.derivative_basis
        phi_x_at_x = x_basis.matrix_on_grid(x_basis.axes)
        dphi_z_at_x = z_basis.matrix_on_grid(
            x_basis.axes, self._derivative_orders(1)
        )
        dphi_x_at_z = x_basis.matrix_on_grid(
            z_basis.axes, self._derivative_orders(1)
        )
        d2phi_z_at_z = z_basis.matrix_on_grid(
            z_basis.axes, self._derivative_orders(2)
        )
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
        return C, B

    def _posterior_moments(self, precision, eta):
        total = self.function_size + self.derivative_size
        precision = np.asarray(precision, dtype=float).reshape(-1)
        eta = np.asarray(eta, dtype=float).reshape(-1)
        if precision.size != total or eta.size != total:
            raise ValueError("site size mismatch")

        system = (self.C + diags(precision, format="csc") @ self.B).tocsc()
        system.sum_duplicates()
        system.eliminate_zeros()
        try:
            lu = splu(system, permc_spec="COLAMD")
        except RuntimeError as error:
            raise np.linalg.LinAlgError("sparse posterior factorization failed") from error

        weights = np.asarray(lu.solve(eta), dtype=float).reshape(-1)
        mean = np.asarray(self.B @ weights, dtype=float).reshape(-1)
        n = self.function_size
        m = self.derivative_size
        derivative_variance = np.empty(m, dtype=float)
        derivative_rows = self.B[n:, :].tocsr()

        for start in range(0, m, self.variance_batch_size):
            stop = min(start + self.variance_batch_size, m)
            width = stop - start
            rhs = np.zeros((total, width), dtype=float)
            rhs[n + np.arange(start, stop), np.arange(width)] = 1.0
            inverse_columns = np.asarray(lu.solve(rhs), dtype=float)
            block = np.asarray(
                derivative_rows[start:stop, :] @ inverse_columns,
                dtype=float,
            )
            derivative_variance[start:stop] = np.diag(block)

        derivative_variance = np.maximum(derivative_variance, 1.0e-14)
        return mean, derivative_variance, weights

    def fit(self, y):
        y = np.asarray(y, dtype=float)
        if y.shape == self.function_shape:
            y = y.reshape(-1, order="C")
        else:
            y = y.reshape(-1)
        if y.size != self.function_size or np.any(~np.isfinite(y)):
            raise ValueError("invalid training values")

        n = self.function_size
        m = self.derivative_size
        total = n + m
        observation_precision = np.zeros(n, dtype=float)
        observation_precision[self.observation_mask] = (
            1.0 / self.noise_variance[self.observation_mask]
        )

        fixed_precision = np.full(total, self.ridge_precision, dtype=float)
        fixed_precision[:n] += self.function_regularization_precision
        fixed_precision[n:] += self.derivative_regularization_precision
        fixed_precision[:n] += observation_precision
        fixed_eta = np.zeros(total, dtype=float)
        fixed_eta[:n] = observation_precision * y
        site_tau = np.zeros(m, dtype=float)
        site_eta = np.zeros(m, dtype=float)
        self.converged = False

        for iteration in range(1, self.max_iter + 1):
            precision = fixed_precision.copy()
            precision[n:] += site_tau
            eta = fixed_eta.copy()
            eta[n:] += site_eta
            mean, derivative_variance, _ = self._posterior_moments(precision, eta)
            derivative_mean = mean[n:]
            cavity_precision = 1.0 / derivative_variance - site_tau
            cavity_eta = derivative_mean / derivative_variance - site_eta
            valid = cavity_precision > 1.0e-14
            proposed_tau = site_tau.copy()
            proposed_eta = site_eta.copy()

            if np.any(valid):
                cavity_variance = 1.0 / cavity_precision[valid]
                cavity_mean = cavity_eta[valid] / cavity_precision[valid]
                denominator = np.sqrt(cavity_variance + self.probit_scale**2)
                z = self.monotone_sign * cavity_mean / denominator
                ratio = _inverse_mills_ratio(z)
                tilted_mean = (
                    cavity_mean
                    + self.monotone_sign
                    * cavity_variance
                    / denominator
                    * ratio
                )
                tilted_variance = cavity_variance - (
                    cavity_variance**2
                    / (cavity_variance + self.probit_scale**2)
                    * ratio
                    * (ratio + z)
                )
                tilted_variance = np.maximum(tilted_variance, 1.0e-12)
                new_tau = 1.0 / tilted_variance - cavity_precision[valid]
                new_eta = tilted_mean / tilted_variance - cavity_eta[valid]
                negative = new_tau < 0.0
                new_tau[negative] = 0.0
                new_eta[negative] = 0.0
                proposed_tau[valid] = new_tau
                proposed_eta[valid] = new_eta

            updated_tau = (
                (1.0 - self.damping) * site_tau
                + self.damping * proposed_tau
            )
            updated_eta = (
                (1.0 - self.damping) * site_eta
                + self.damping * proposed_eta
            )
            max_change = max(
                np.max(np.abs(updated_tau - site_tau), initial=0.0),
                np.max(np.abs(updated_eta - site_eta), initial=0.0),
            )
            site_tau = updated_tau
            site_eta = updated_eta
            self.iterations = iteration
            self.final_site_change = float(max_change)

            if self.verbose:
                violations = np.mean(
                    self.monotone_sign * derivative_mean < 0.0
                )
                print(
                    f"EP {iteration:03d} | change {max_change:.3e} | "
                    f"violations {violations:.2%}"
                )

            if max_change < self.tolerance:
                self.converged = True
                break
        
        if not self.converged:
            raise RuntimeError(f"EP did not converge; final change = {self.final_site_change:.3e}")

        precision = fixed_precision.copy()
        precision[n:] += site_tau
        eta = fixed_eta.copy()
        eta[n:] += site_eta
        mean, derivative_variance, weights = self._posterior_moments(
            precision, eta
        )
        self.posterior_packet_weights = weights
        self.posterior_mean_function = mean[:n].copy()
        self.posterior_mean_derivative = mean[n:].copy()
        self.posterior_variance_derivative = derivative_variance.copy()
        self.site_precision_derivative = site_tau.copy()
        self.site_eta_derivative = site_eta.copy()
        return self

    def _cross_packet_matrix(self, points, derivative_dimension=None):
        if derivative_dimension is None:
            left_orders = (0,) * self.dimension
            right_orders = self._derivative_orders(1)
        else:
            derivative_dimension = int(derivative_dimension)
            if not 0 <= derivative_dimension < self.dimension:
                raise ValueError("invalid derivative dimension")
            left_orders = [0] * self.dimension
            left_orders[derivative_dimension] = 1
            right_orders = [0] * self.dimension
            if derivative_dimension == self.derivative_dim:
                right_orders[self.derivative_dim] = 2
            else:
                right_orders[derivative_dimension] = 1
                right_orders[self.derivative_dim] = 1
            left_orders = tuple(left_orders)
            right_orders = tuple(right_orders)

        left = self.function_basis.matrix(points, left_orders)
        right = self.derivative_basis.matrix(points, right_orders)
        return self.variance * hstack([left, -right], format="csc")

    def predict(self, points):
        if self.posterior_packet_weights is None:
            raise RuntimeError("fit must be called first")
        matrix = self._cross_packet_matrix(points)
        return np.asarray(
            matrix @ self.posterior_packet_weights, dtype=float
        ).reshape(-1)

    def predict_derivative(self, points, derivative_dimension):
        if self.posterior_packet_weights is None:
            raise RuntimeError("fit must be called first")
        matrix = self._cross_packet_matrix(points, derivative_dimension)
        return np.asarray(
            matrix @ self.posterior_packet_weights, dtype=float
        ).reshape(-1)

def _grid_inputs(s_train, T_train, q_train, noise_std, observation_mask):
    s_array = np.asarray(s_train, dtype=float)
    T_array = np.asarray(T_train, dtype=float)
    q_array = np.asarray(q_train, dtype=float)

    if q_array.ndim == 2:
        s_axis = _validate_axis(s_array, "s axis")
        T_axis = _validate_axis(T_array, "temperature axis")

        if q_array.shape != (s_axis.size, T_axis.size):
            raise ValueError("q_grid shape does not match the coordinate axes")

        q_grid = q_array.copy()

        if observation_mask is None:
            mask_grid = np.isfinite(q_grid)
        else:
            mask_grid = np.asarray(observation_mask, dtype=bool)
            if mask_grid.shape != q_grid.shape:
                if mask_grid.size == q_grid.size:
                    mask_grid = mask_grid.reshape(q_grid.shape, order="C")
                else:
                    raise ValueError("observation_mask does not match q_grid")
            mask_grid &= np.isfinite(q_grid)

        if not np.any(mask_grid):
            raise ValueError("observation_mask contains no finite observations")

        fill_value = float(np.mean(q_grid[mask_grid]))
        q_grid[~np.isfinite(q_grid)] = fill_value

        if noise_std is None:
            sigma_grid = None
        else:
            sigma = np.asarray(noise_std, dtype=float)
            if sigma.size == 1:
                sigma_grid = np.full(q_grid.shape, float(sigma.reshape(-1)[0]))
            elif sigma.shape == q_grid.shape:
                sigma_grid = sigma.copy()
            elif sigma.size == q_grid.size:
                sigma_grid = sigma.reshape(q_grid.shape, order="C").copy()
            else:
                raise ValueError("noise_std does not match q_grid")

            invalid_observed_noise = mask_grid & (
                ~np.isfinite(sigma_grid) | (sigma_grid < 0.0)
            )
            if np.any(invalid_observed_noise):
                raise ValueError("observed noise values must be finite and nonnegative")
            sigma_grid[~mask_grid] = 0.0

        return s_axis, T_axis, q_grid, sigma_grid, mask_grid

    s = s_array.reshape(-1)
    T = T_array.reshape(-1)
    q = q_array.reshape(-1)

    if not (s.size == T.size == q.size) or s.size == 0:
        raise ValueError("s_train, T_train, and q_train must have equal lengths")
    if np.any(~np.isfinite(s)) or np.any(~np.isfinite(T)) or np.any(~np.isfinite(q)):
        raise ValueError("training data must be finite")

    s_axis = np.unique(s)
    T_axis = np.unique(T)
    grid_size = s_axis.size * T_axis.size
    s_index = np.searchsorted(s_axis, s)
    T_index = np.searchsorted(T_axis, T)
    flat_index = s_index * T_axis.size + T_index

    if np.unique(flat_index).size != flat_index.size:
        raise ValueError("duplicate Cartesian training points")

    q_flat = np.full(grid_size, np.nan)
    q_flat[flat_index] = q
    q_grid = q_flat.reshape((s_axis.size, T_axis.size), order="C")
    finite_grid = np.isfinite(q_grid)

    if observation_mask is None:
        mask_grid = finite_grid
    else:
        supplied_mask = np.asarray(observation_mask, dtype=bool)
        if supplied_mask.size == q.size:
            mask_flat = np.zeros(grid_size, dtype=bool)
            mask_flat[flat_index] = supplied_mask.reshape(-1)
            mask_grid = mask_flat.reshape(q_grid.shape, order="C")
        elif supplied_mask.size == grid_size:
            mask_grid = supplied_mask.reshape(q_grid.shape, order="C")
        else:
            raise ValueError("invalid observation_mask")
        mask_grid &= finite_grid

    if not np.any(mask_grid):
        raise ValueError("observation_mask contains no finite observations")

    fill_value = float(np.mean(q_grid[mask_grid]))
    q_grid[~finite_grid] = fill_value

    if noise_std is None:
        sigma_grid = None
    else:
        sigma = np.asarray(noise_std, dtype=float).reshape(-1)
        if sigma.size == 1:
            sigma_grid = np.full(q_grid.shape, float(sigma[0]))
        elif sigma.size == q.size:
            sigma_flat = np.full(grid_size, np.nan)
            sigma_flat[flat_index] = sigma
            sigma_grid = sigma_flat.reshape(q_grid.shape, order="C")
        elif sigma.size == grid_size:
            sigma_grid = sigma.reshape(q_grid.shape, order="C").copy()
        else:
            raise ValueError("noise_std must be scalar or match the training rows")

        invalid_observed_noise = mask_grid & (
            ~np.isfinite(sigma_grid) | (sigma_grid < 0.0)
        )
        if np.any(invalid_observed_noise):
            raise ValueError("observed noise values must be finite and nonnegative")
        sigma_grid[~mask_grid] = 0.0

    return s_axis, T_axis, q_grid, sigma_grid, mask_grid

def _dense_product_covariance(axes, lengthscales, nu, variance):
    factors = []
    for axis, lengthscale in zip(axes, lengthscales):
        kernel = MaternHalfInteger1D(nu, lengthscale)
        factors.append(kernel.covariance(axis, axis))
    covariance = factors[0]
    for factor in factors[1:]:
        covariance = np.kron(covariance, factor)
    return float(variance) * covariance


def _select_lengthscales(
    axes,
    y_grid,
    noise_variance,
    observation_mask,
    nu,
    variance,
    candidates,
):
    candidates = tuple(float(value) for value in candidates)
    if not candidates or any(
        not np.isfinite(value) or value <= 0.0 for value in candidates
    ):
        raise ValueError("invalid lengthscale candidates")

    y = np.asarray(y_grid, dtype=float).reshape(-1, order="C")
    noise = np.asarray(noise_variance, dtype=float).reshape(-1, order="C")
    observed = np.flatnonzero(
        np.asarray(observation_mask, dtype=bool).reshape(-1, order="C")
    )
    if observed.size < 4:
        return (1.0,) * len(axes)

    validation = observed[::5]
    training = np.setdiff1d(observed, validation, assume_unique=True)
    if validation.size == 0 or training.size < 3:
        validation = observed[-max(1, observed.size // 5):]
        training = np.setdiff1d(observed, validation, assume_unique=True)

    best_score = np.inf
    best = None
    for values in product(candidates, repeat=len(axes)):
        covariance = _dense_product_covariance(axes, values, nu, variance)
        K_train = covariance[np.ix_(training, training)]
        K_train = K_train + np.diag(noise[training])
        scale = max(float(np.max(np.diag(K_train))), 1.0)
        solved = False
        for jitter in (0.0, 1.0e-12 * scale, 1.0e-10 * scale, 1.0e-8 * scale):
            try:
                L = np.linalg.cholesky(K_train + jitter * np.eye(training.size))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y[training]))
                solved = True
                break
            except np.linalg.LinAlgError:
                continue
        if not solved:
            continue
        prediction = covariance[np.ix_(validation, training)] @ alpha
        score = float(np.sqrt(np.mean((prediction - y[validation]) ** 2)))
        if score < best_score:
            best_score = score
            best = tuple(values)

    if best is None:
        raise RuntimeError("lengthscale search failed")
    return best

class MonotoneGPKPFluxST:
    def __init__(
        self,
        s_train,
        T_train,
        q_train,
        noise_std=None,
        observation_mask=None,
        noise_is_relative=True,
        learn_neg_flux=True,
        nu=2.5,
        lengthscale="auto",
        lengthscale_candidates=(0.25, 0.5, 1.0, 2.0, 4.0),
        variance=1.0,
        n_virtual_per_axis=7,
        probit_nu=5.0e-2,
        ep_max_iter=100,
        ep_damping=0.4,
        ep_tol=1.0e-6,
        jitter=1.0e-10,
        use_tikhonov=False,
        function_regularization=0.0,
        derivative_regularization=0.0,
        variance_batch_size=32,
        prediction_batch_size=4096,
        verbose=False,
    ):
        self.learn_neg_flux = bool(learn_neg_flux)
        self.noise_is_relative = bool(noise_is_relative)
        self.nu = float(nu)
        self.lengthscale = lengthscale
        self.lengthscale_candidates = tuple(lengthscale_candidates)
        self.variance = float(variance)
        self.n_virtual_per_axis = int(n_virtual_per_axis)
        self.probit_nu = float(probit_nu)
        self.ep_max_iter = int(ep_max_iter)
        self.ep_damping = float(ep_damping)
        self.ep_tol = float(ep_tol)
        self.jitter = float(jitter)
        self.use_tikhonov = bool(use_tikhonov)
        self.function_regularization = float(function_regularization)
        self.derivative_regularization = float(derivative_regularization)
        self.variance_batch_size = int(variance_batch_size)
        self.prediction_batch_size = int(prediction_batch_size)
        self.verbose = bool(verbose)
        self.model = None
        self.selected_lengthscales = None
        self.ep_damping_used = None
        self._validate_configuration()
        self.fit(s_train, T_train, q_train, noise_std, observation_mask)

    def _validate_configuration(self):
        order = half_integer_order(self.nu)
        if order < 1:
            raise ValueError("nu must be at least 3/2")
        if isinstance(self.lengthscale, str):
            if self.lengthscale != "auto":
                raise ValueError("lengthscale must be positive or 'auto'")
        else:
            _lengthscale_tuple(self.lengthscale, 2)
        if not np.isfinite(self.variance) or self.variance <= 0.0:
            raise ValueError("variance must be positive")
        if self.n_virtual_per_axis < 2 * order + 3:
            raise ValueError("too few virtual points")
        if not np.isfinite(self.probit_nu) or self.probit_nu <= 0.0:
            raise ValueError("probit_nu must be positive")
        if self.ep_max_iter < 1 or not 0.0 < self.ep_damping <= 1.0:
            raise ValueError("invalid EP controls")
        if self.ep_tol <= 0.0 or self.jitter < 0.0:
            raise ValueError("invalid numerical controls")
        if self.variance_batch_size < 1 or self.prediction_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        if any(
            not np.isfinite(value) or value < 0.0
            for value in (
                self.function_regularization,
                self.derivative_regularization,
            )
        ):
            raise ValueError("regularization must be nonnegative")
    def _noise_variance(self, sigma_grid):
        if sigma_grid is None:
            absolute_sigma = np.zeros(self.q_grid.shape, dtype=float)
        else:
            sigma_grid = np.asarray(sigma_grid, dtype=float)
            if np.any(~np.isfinite(sigma_grid)) or np.any(sigma_grid < 0.0):
                raise ValueError("noise_std must be finite and nonnegative")
            absolute_sigma = sigma_grid.copy()
            if self.noise_is_relative:
                absolute_sigma *= np.maximum(1.0, np.abs(self.q_grid))
        return np.maximum((absolute_sigma / self.y_scale) ** 2, 1.0e-10)

    def _fit_model(self, function_axes, derivative_axes, y, noise, mask, damping):
        return MonotoneKernelPacketGP(
            function_axes,
            derivative_axes,
            self.nu,
            self.selected_lengthscales,
            self.variance,
            noise,
            mask,
            0,
            1 if self.learn_neg_flux else -1,
            self.probit_nu,
            self.ep_max_iter,
            damping,
            self.ep_tol,
            self.jitter,
            self.use_tikhonov,
            self.function_regularization,
            self.derivative_regularization,
            self.variance_batch_size,
            self.verbose,
        ).fit(y)

    def fit(self, s_train, T_train, q_train, noise_std=None, observation_mask=None):
        s_axis, T_axis, q_grid, sigma_grid, mask_grid = _grid_inputs(
            s_train,
            T_train,
            q_train,
            noise_std,
            observation_mask,
        )

        minimum = 2 * half_integer_order(self.nu) + 3
        if s_axis.size < minimum or T_axis.size < minimum:
            raise ValueError("too few training coordinates per axis")

        latent_grid = -q_grid if self.learn_neg_flux else q_grid
        self.s_axis = s_axis.copy()
        self.T_axis = T_axis.copy()
        self.q_grid = q_grid.copy()
        self.observation_mask = mask_grid.copy()
        self.s_mean = float(np.mean(s_axis))
        self.s_scale = float(np.std(s_axis))
        self.T_mean = float(np.mean(T_axis))
        self.T_scale = float(np.std(T_axis))
        observed_latent = latent_grid[mask_grid]
        self.y_mean = float(np.mean(observed_latent))
        self.y_scale = float(np.std(observed_latent))

        for scale in (self.s_scale, self.T_scale, self.y_scale):
            if not np.isfinite(scale) or scale <= np.finfo(float).eps:
                raise ValueError("training data must vary in s, T, and q")

        s_standardized = (s_axis - self.s_mean) / self.s_scale
        T_standardized = (T_axis - self.T_mean) / self.T_scale
        y_standardized = (latent_grid - self.y_mean) / self.y_scale
        noise_variance = self._noise_variance(sigma_grid)

        if self.lengthscale == "auto":
            self.selected_lengthscales = _select_lengthscales(
                (s_standardized, T_standardized),
                y_standardized,
                noise_variance,
                mask_grid,
                self.nu,
                self.variance,
                self.lengthscale_candidates,
            )
        else:
            self.selected_lengthscales = _lengthscale_tuple(self.lengthscale, 2)

        s_virtual = np.linspace(s_axis[0], s_axis[-1], self.n_virtual_per_axis)
        T_virtual = np.linspace(T_axis[0], T_axis[-1], self.n_virtual_per_axis)
        s_virtual = (s_virtual - self.s_mean) / self.s_scale
        T_virtual = (T_virtual - self.T_mean) / self.T_scale

        failures = []
        damping_values = []
        damping = self.ep_damping
        while damping >= max(0.05, self.ep_damping / 8.0):
            if not any(np.isclose(damping, value) for value in damping_values):
                damping_values.append(damping)
            damping *= 0.5

        for damping in damping_values:
            try:
                self.model = self._fit_model(
                    (s_standardized, T_standardized),
                    (s_virtual, T_virtual),
                    y_standardized,
                    noise_variance,
                    mask_grid,
                    damping,
                )
                self.ep_damping_used = damping
                return self
            except (RuntimeError, np.linalg.LinAlgError) as error:
                failures.append(str(error))

        raise RuntimeError("EP failed: " + " | ".join(failures))

    def evaluate(self, s_q, T_q):
        if self.model is None:
            raise RuntimeError("provider is not fitted")

        s = np.atleast_1d(np.asarray(s_q, dtype=float))
        T = np.atleast_1d(np.asarray(T_q, dtype=float))
        try:
            s, T = np.broadcast_arrays(s, T)
        except ValueError as error:
            raise ValueError("s_q and T_q are not broadcast-compatible") from error
        if np.any(~np.isfinite(s)) or np.any(~np.isfinite(T)):
            raise ValueError("query points must be finite")

        shape = s.shape
        points = np.column_stack(
            [
                (s.reshape(-1) - self.s_mean) / self.s_scale,
                (T.reshape(-1) - self.T_mean) / self.T_scale,
            ]
        )
        count = points.shape[0]
        f = np.empty(count)
        df_ds = np.empty(count)
        df_dT = np.empty(count)

        for start in range(0, count, self.prediction_batch_size):
            stop = min(start + self.prediction_batch_size, count)
            batch = points[start:stop]
            f[start:stop] = self.model.predict(batch)
            df_ds[start:stop] = self.model.predict_derivative(batch, 0)
            df_dT[start:stop] = self.model.predict_derivative(batch, 1)

        f = self.y_mean + self.y_scale * f
        df_ds *= self.y_scale / self.s_scale
        df_dT *= self.y_scale / self.T_scale
        sign = -1.0 if self.learn_neg_flux else 1.0
        return (
            (sign * f).reshape(shape),
            (sign * df_ds).reshape(shape),
            (sign * df_dT).reshape(shape),
        )


__all__ = ["MonotoneGPKPFluxST"]