# No-Noise No-Smoothing Baseline Tables

Source CSVs: `Results/no_noise_no_smoothing_baselines/*.csv`

Methods: Analytic, FiniteDiff, CubicSpline, PCHIP, RBF, SavGol, KISS-GP, MLP. No `+reg=...` variants, no RFF, and no smoothing-style FDIntegrated/FDMatern/Smooth+PCHIP methods.

## Linear (No-noise)

\begin{table}[h]
\centering
\scriptsize
\caption{Linear (No-noise) Data Accuracy}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
method & q RMSE & q/noise & $q_s$ RMSE & $q_s$/noise & $q_T$ RMSE & $q_T$/noise \\
\midrule
Analytic         & 3.0969e-16 & -- & 0.0000e+00 & -- & 0.0000e+00 & -- \\
SavGol           & 2.0556e-15 & -- & 1.4723e-15 & -- & 3.1827e-15 & -- \\
KISS-GP          & 9.0675e-02 & -- & 2.1693e-01 & -- & 1.7054e-01 & -- \\
MLP              & 1.2674e-01 & -- & 3.2327e-01 & -- & 2.8922e-01 & -- \\
RBF              & 6.2419e-01 & -- & 1.3147e+00 & -- & 1.9801e+00 & -- \\
CubicSpline      & 2.1670e-01 & -- & 7.0608e-01 & -- & 1.8207e+00 & -- \\
PCHIP            & -- & -- & -- & -- & -- & -- \\
FiniteDiff       & 1.8332e-01 & -- & 2.4158e-01 & -- & 5.3441e-01 & -- \\
\bottomrule
\end{tabular}%
}
\end{table}

\begin{table}[h]
\centering
\scriptsize
\caption{Linear (No-noise) Newton Solve Results}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
method & status & build\_s & newton\_steps & flux\_calls & rel\_solution\_err & solve\_total\_s & flux\_eval\_s & avg\_flux\_eval\_us \\
\midrule
Analytic         & converged & 6.5087e-02 & 1.0000e+00 & 2.4000e+02 & 0.0000e+00 & 8.5129e-03 & 2.9150e-04 & 1.2146e+00 \\
SavGol           & converged & 6.5656e-02 & 1.0000e+00 & 2.4000e+02 & 1.9294e-16 & 4.4947e-02 & 3.4325e-02 & 1.4302e+02 \\
KISS-GP          & converged & 3.6669e-01 & 3.0000e+00 & 5.6000e+02 & 5.6013e-03 & 1.7209e+00 & 1.6783e+00 & 2.9969e+03 \\
MLP              & converged & 4.0750e+00 & 3.0000e+00 & 5.6000e+02 & 1.1149e-02 & 5.5425e+00 & 5.4960e+00 & 9.8143e+03 \\
RBF              & converged & 6.7326e-02 & 3.0000e+00 & 5.6000e+02 & 3.1279e-02 & 3.2737e-02 & 1.2234e-02 & 2.1846e+01 \\
CubicSpline      & converged & 6.5617e-02 & 9.0000e+00 & 2.0000e+03 & 1.8356e-02 & 8.6577e-02 & 1.4834e-02 & 7.4170e+00 \\
PCHIP            & converged & 6.8061e-02 & 1.7000e+01 & 3.2000e+03 & 1.3714e-02 & 1.2044e+00 & 1.0663e+00 & 3.3322e+02 \\
FiniteDiff       & not converged & 6.5123e-02 & -- & 6.4800e+03 & -- & 8.1091e-01 & 5.6381e-01 & 8.7008e+01 \\
\bottomrule
\end{tabular}%
}
\end{table}

\begin{table}[h]
\centering
\scriptsize
\caption{Linear (No-noise) Data Physical Correctness}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccc}
\toprule
method & entropy\_violation\_\% & worst\_entropy\_violation & deriv\_violation\_\% & worst\_deriv\_violation \\
\midrule
Analytic         & 0.0000e+00 & 0.0000e+00 & 0.0000e+00 & 0.0000e+00 \\
SavGol           & 0.0000e+00 & 0.0000e+00 & 0.0000e+00 & 0.0000e+00 \\
KISS-GP          & 0.0000e+00 & 0.0000e+00 & 0.0000e+00 & 0.0000e+00 \\
MLP              & 8.0000e-01 & 2.9931e-03 & 0.0000e+00 & 0.0000e+00 \\
RBF              & 8.4000e+00 & 9.1448e-02 & 1.8200e+01 & 2.6231e+00 \\
CubicSpline      & 2.0000e+00 & 6.9044e-02 & 2.0000e+00 & 1.1005e+00 \\
PCHIP            & 2.0000e+00 & 5.9459e-02 & 0.0000e+00 & 0.0000e+00 \\
FiniteDiff       & 2.0000e+00 & 3.2019e-02 & 0.0000e+00 & 0.0000e+00 \\
\bottomrule
\end{tabular}%
}
\end{table}

## Nonlinear (No-noise)

\begin{table}[h]
\centering
\scriptsize
\caption{Nonlinear (No-noise) Data Accuracy}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
method & q RMSE & q/noise & $q_s$ RMSE & $q_s$/noise & $q_T$ RMSE & $q_T$/noise \\
\midrule
Analytic         & 1.2962e-15 & -- & 0.0000e+00 & -- & 0.0000e+00 & -- \\
MLP              & 3.5777e-01 & -- & 1.2080e+00 & -- & 1.4150e+00 & -- \\
KISS-GP          & 3.1269e-01 & -- & 1.0137e+00 & -- & 8.9430e-01 & -- \\
SavGol           & 1.0844e-02 & -- & 1.1573e-01 & -- & 3.7609e-02 & -- \\
CubicSpline      & 7.8880e-01 & -- & 2.5036e+00 & -- & 4.5857e+00 & -- \\
PCHIP            & -- & -- & -- & -- & -- & -- \\
FiniteDiff       & 7.0599e-01 & -- & 9.4205e-01 & -- & 1.4498e+00 & -- \\
RBF              & 2.7494e+00 & -- & 5.0420e+00 & -- & 9.1676e+00 & -- \\
\bottomrule
\end{tabular}%
}
\end{table}

\begin{table}[h]
\centering
\scriptsize
\caption{Nonlinear (No-noise) Newton Solve Results}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
method & status & build\_s & newton\_steps & flux\_calls & rel\_solution\_err & solve\_total\_s & flux\_eval\_s & avg\_flux\_eval\_us \\
\midrule
Analytic         & converged & 6.5668e-02 & 4.0000e+00 & 7.2000e+02 & 0.0000e+00 & 2.6646e-02 & 8.7690e-04 & 1.2179e+00 \\
MLP              & converged & 5.7077e+00 & 4.0000e+00 & 7.2000e+02 & 1.5708e-02 & 6.8998e+00 & 6.8449e+00 & 9.5068e+03 \\
KISS-GP          & converged & 8.0496e-01 & 4.0000e+00 & 7.2000e+02 & 2.2230e-02 & 2.2306e+00 & 2.1749e+00 & 3.0208e+03 \\
SavGol           & converged & 6.6513e-02 & 6.0000e+00 & 1.0400e+03 & 9.8600e-05 & 1.9509e-01 & 1.5075e-01 & 1.4496e+02 \\
CubicSpline      & converged & 7.0047e-02 & 7.0000e+00 & 1.4400e+03 & 1.0461e-02 & 6.9163e-02 & 1.1454e-02 & 7.9544e+00 \\
PCHIP            & converged & 7.3543e-02 & 1.8000e+01 & 3.2800e+03 & 8.0701e-03 & 1.3139e+00 & 1.1623e+00 & 3.5436e+02 \\
FiniteDiff       & converged & 6.5136e-02 & 2.0000e+01 & 3.3600e+03 & 6.0154e-03 & 4.3667e-01 & 3.0086e-01 & 8.9542e+01 \\
RBF              & not converged & 6.9865e-02 & -- & 8.1600e+03 & -- & 4.8243e-01 & 1.7984e-01 & 2.2039e+01 \\
\bottomrule
\end{tabular}%
}
\end{table}

\begin{table}[h]
\centering
\scriptsize
\caption{Nonlinear (No-noise) Data Physical Correctness}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccc}
\toprule
method & entropy\_violation\_\% & worst\_entropy\_violation & deriv\_violation\_\% & worst\_deriv\_violation \\
\midrule
Analytic         & 0.0000e+00 & 0.0000e+00 & 0.0000e+00 & 0.0000e+00 \\
MLP              & 1.2000e+00 & 2.3913e-02 & 2.0000e-01 & 2.4793e-01 \\
KISS-GP          & 0.0000e+00 & 0.0000e+00 & 0.0000e+00 & 0.0000e+00 \\
SavGol           & 0.0000e+00 & 0.0000e+00 & 0.0000e+00 & 0.0000e+00 \\
CubicSpline      & 1.8000e+00 & 1.0735e-01 & 3.0000e+00 & 7.6152e+00 \\
PCHIP            & 1.8000e+00 & 1.1290e-01 & 0.0000e+00 & 0.0000e+00 \\
FiniteDiff       & 1.6000e+00 & 7.6708e-02 & 0.0000e+00 & 0.0000e+00 \\
RBF              & 2.5000e+01 & 2.6888e+00 & 2.7600e+01 & 6.8628e+00 \\
\bottomrule
\end{tabular}%
}
\end{table}
