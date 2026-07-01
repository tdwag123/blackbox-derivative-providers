# 1D diffusion datasets

These CSVs are small test datasets for the 1D constitutive law

```text
q_true = -(k_0(1 + alpha T^2) + beta s^2)s
```

Each row is one local state, like a quadrature point. Columns:

```text
s      = dT/dx
T      = temperature
x      = position in the 1D domain
k_0, alpha, beta = fixed material parameters for that file
sigma  = Gaussian noise level
q_true = clean physical flux
q_noisy = noisy measured flux
a_true = dq/ds
b_true = dq/dT
```

Intended use:

```text
train on:      s, T -> q_noisy
validate with: q_true, a_true, b_true
```

The three files are just different cases: nonlinear high noise, nonlinear low noise, and linear medium noise.
