# Best-Newton LaTeX Tables

Source CSVs: `Results/bello_6/*.csv`

For each dataset and method family, the row shown is the successful variant with the fewest Newton steps. If a family has no successful Newton solve, the best available failed row is included as `not converged`.

## Linear (Medium-noise)

\begin{table}[h]
\centering
\scriptsize
\caption{Linear (Medium-noise) Data Accuracy, best Newton variant per method family}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
method & q RMSE & q/noise & $q_s$ RMSE & $q_s$/noise & $q_T$ RMSE & $q_T$/noise \\
\midrule
FDIntegratedEpanechnikov+gradient:0.3 & 1.9002e-01 & 1.5203e+00 & 9.1651e-02 & 3.7803e-01 & 4.4869e-02 & 9.2359e-02 \\
RFF+laplacian:0.01             & 2.1012e-01 & 1.8052e+00 & 2.5731e-01 & 1.0613e+00 & 5.0816e-01 & 1.0460e+00 \\
KISS-GP                        & 1.6846e-01 & 1.2366e+00 & 2.3633e-01 & 9.7475e-01 & 1.8438e-01 & 3.7953e-01 \\
MLP                            & 1.7865e-01 & 1.7272e+00 & 3.2467e-01 & 1.3392e+00 & 2.7681e-01 & 5.6978e-01 \\
RBF                            & 6.6230e-01 & 7.3389e+00 & 1.2606e+00 & 5.1996e+00 & 2.1309e+00 & 4.3862e+00 \\
Smooth+PCHIP+gradient:0.3      & -- & -- & -- & -- & -- & -- \\
PCHIP+gradient:0.3             & -- & -- & -- & -- & -- & -- \\
FDMatern52+gradient:0.3        & 2.2721e-01 & 1.6815e+00 & 1.9815e-01 & 8.1731e-01 & 1.7916e-01 & 3.6879e-01 \\
FiniteDiff+laplacian:0.1       & 2.1321e-01 & 1.8393e+00 & 2.1459e-01 & 8.8508e-01 & 3.4782e-01 & 7.1596e-01 \\
SavGol                         & 1.3200e-01 & 1.0083e+00 & 9.3230e-02 & 3.8454e-01 & 2.2523e-01 & 4.6361e-01 \\
\bottomrule
\end{tabular}%
}
\end{table}

\begin{table}[h]
\centering
\scriptsize
\caption{Linear (Medium-noise) Newton Solve Results, best Newton variant per method family}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
method & status & build\_s & newton\_steps & flux\_calls & rel\_solution\_err & solve\_total\_s & flux\_eval\_s & avg\_flux\_eval\_us \\
\midrule
FDIntegratedEpanechnikov+gradient:0.3 & converged & 8.0125e-02 & 3.0000e+00 & 5.6000e+02 & 3.7748e-03 & 5.5462e-02 & 3.4271e-02 & 6.1199e+01 \\
RFF+laplacian:0.01             & converged & 1.0135e-01 & 3.0000e+00 & 5.6000e+02 & 4.2817e-03 & 1.2555e-01 & 1.0049e-01 & 1.7944e+02 \\
KISS-GP                        & converged & 3.5971e-01 & 3.0000e+00 & 5.6000e+02 & 5.9026e-03 & 1.7135e+00 & 1.6729e+00 & 2.9873e+03 \\
MLP                            & converged & 4.0910e+00 & 3.0000e+00 & 5.6000e+02 & 1.2382e-02 & 5.4441e+00 & 5.4012e+00 & 9.6450e+03 \\
RBF                            & converged & 6.9333e-02 & 3.0000e+00 & 5.6000e+02 & 3.3799e-02 & 3.2504e-02 & 1.2133e-02 & 2.1666e+01 \\
Smooth+PCHIP+gradient:0.3      & converged & 7.1843e-02 & 4.0000e+00 & 7.2000e+02 & 7.1423e-03 & 2.7640e-01 & 2.4452e-01 & 3.3961e+02 \\
PCHIP+gradient:0.3             & converged & 6.9372e-02 & 6.0000e+00 & 1.0400e+03 & 5.3765e-03 & 3.9517e-01 & 3.5042e-01 & 3.3694e+02 \\
FDMatern52+gradient:0.3        & converged & 7.5919e-02 & 6.0000e+00 & 1.0400e+03 & 7.1840e-03 & 1.0379e-01 & 6.3610e-02 & 6.1164e+01 \\
FiniteDiff+laplacian:0.1       & converged & 6.9881e-02 & 1.0000e+01 & 1.6800e+03 & 7.6079e-03 & 2.1149e-01 & 1.4763e-01 & 8.7872e+01 \\
SavGol                         & not converged & 6.7102e-02 & -- & 3.6560e+04 & -- & 6.7052e+00 & 5.1906e+00 & 1.4197e+02 \\
\bottomrule
\end{tabular}%
}
\end{table}

\begin{table}[h]
\centering
\scriptsize
\caption{Linear (Medium-noise) Data Physical Correctness, best Newton variant per method family}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccc}
\toprule
method & entropy\_violation\_\% & worst\_entropy\_violation & deriv\_violation\_\% & worst\_deriv\_violation \\
\midrule
FDIntegratedEpanechnikov+gradient:0.3 & 1.0000e+00 & 8.5349e-04 & 0.0000e+00 & 0.0000e+00 \\
RFF+laplacian:0.01             & 4.0000e-01 & 3.0201e-04 & 0.0000e+00 & 0.0000e+00 \\
KISS-GP                        & 0.0000e+00 & 0.0000e+00 & 0.0000e+00 & 0.0000e+00 \\
MLP                            & 6.0000e-01 & 2.0451e-03 & 0.0000e+00 & 0.0000e+00 \\
RBF                            & 9.2000e+00 & 8.9569e-02 & 1.8000e+01 & 2.1265e+00 \\
Smooth+PCHIP+gradient:0.3      & 2.0000e-01 & 2.0474e-04 & 0.0000e+00 & 0.0000e+00 \\
PCHIP+gradient:0.3             & 0.0000e+00 & 0.0000e+00 & 0.0000e+00 & 0.0000e+00 \\
FDMatern52+gradient:0.3        & 2.0000e-01 & 1.9791e-04 & 0.0000e+00 & 0.0000e+00 \\
FiniteDiff+laplacian:0.1       & 0.0000e+00 & 0.0000e+00 & 0.0000e+00 & 0.0000e+00 \\
SavGol                         & 4.0000e-01 & 5.8813e-04 & 0.0000e+00 & 0.0000e+00 \\
\bottomrule
\end{tabular}%
}
\end{table}

## Nonlinear (High-noise)

\begin{table}[h]
\centering
\scriptsize
\caption{Nonlinear (High-noise) Data Accuracy, best Newton variant per method family}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
method & q RMSE & q/noise & $q_s$ RMSE & $q_s$/noise & $q_T$ RMSE & $q_T$/noise \\
\midrule
Smooth+PCHIP+gradient:0.3      & -- & -- & -- & -- & -- & -- \\
RFF+gradient:0.3               & 9.7451e-01 & 1.4605e+00 & 1.2698e+00 & 1.0687e+00 & 1.5239e+00 & 6.3269e-01 \\
MLP                            & 8.5998e-01 & 1.5251e+00 & 1.4467e+00 & 1.2176e+00 & 1.5504e+00 & 6.4369e-01 \\
KISS-GP                        & 7.7675e-01 & 1.3139e+00 & 1.1261e+00 & 9.4778e-01 & 9.3825e-01 & 3.8954e-01 \\
FDIntegratedEpanechnikov+laplacian:0.01 & 1.4814e+00 & 2.4900e+00 & 1.9010e+00 & 1.6000e+00 & 8.4388e-01 & 3.5036e-01 \\
RBF                            & 2.6987e+00 & 1.0347e+01 & 5.3237e+00 & 4.4808e+00 & 8.7212e+00 & 3.6209e+00 \\
SavGol                         & 7.3074e-01 & 1.1722e+00 & 7.5235e-01 & 6.3323e-01 & 9.9959e-01 & 4.1501e-01 \\
PCHIP+gradient:0.3             & -- & -- & -- & -- & -- & -- \\
FDMatern52+laplacian:0.1       & 9.7281e-01 & 1.4056e+00 & 8.5815e-01 & 7.2228e-01 & 7.6169e-01 & 3.1624e-01 \\
FiniteDiff+laplacian:0.1       & 9.4520e-01 & 1.5547e+00 & 1.0590e+00 & 8.9129e-01 & 1.6932e+00 & 7.0298e-01 \\
\bottomrule
\end{tabular}%
}
\end{table}

\begin{table}[h]
\centering
\scriptsize
\caption{Nonlinear (High-noise) Newton Solve Results, best Newton variant per method family}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
method & status & build\_s & newton\_steps & flux\_calls & rel\_solution\_err & solve\_total\_s & flux\_eval\_s & avg\_flux\_eval\_us \\
\midrule
Smooth+PCHIP+gradient:0.3      & converged & 6.9210e-02 & 4.0000e+00 & 7.2000e+02 & 4.3952e-03 & 2.7093e-01 & 2.4005e-01 & 3.3340e+02 \\
RFF+gradient:0.3               & converged & 8.2150e-02 & 4.0000e+00 & 7.2000e+02 & 7.2323e-03 & 1.6329e-01 & 1.3161e-01 & 1.8279e+02 \\
MLP                            & converged & 7.7865e+00 & 4.0000e+00 & 7.2000e+02 & 1.1185e-02 & 7.0954e+00 & 7.0376e+00 & 9.7745e+03 \\
KISS-GP                        & converged & 8.4757e-01 & 4.0000e+00 & 7.2000e+02 & 2.3940e-02 & 2.1960e+00 & 2.1430e+00 & 2.9764e+03 \\
FDIntegratedEpanechnikov+laplacian:0.01 & converged & 7.8244e-02 & 5.0000e+00 & 8.8000e+02 & 2.1326e-03 & 8.6552e-02 & 5.3027e-02 & 6.0258e+01 \\
RBF                            & converged & 6.9758e-02 & 5.0000e+00 & 8.8000e+02 & 4.7911e-02 & 5.0118e-02 & 1.8580e-02 & 2.1113e+01 \\
SavGol                         & converged & 6.6244e-02 & 7.0000e+00 & 1.2000e+03 & 8.5329e-03 & 2.1902e-01 & 1.7005e-01 & 1.4171e+02 \\
PCHIP+gradient:0.3             & converged & 7.1628e-02 & 8.0000e+00 & 1.3600e+03 & 3.5639e-03 & 5.1428e-01 & 4.5520e-01 & 3.3471e+02 \\
FDMatern52+laplacian:0.1       & converged & 7.3751e-02 & 8.0000e+00 & 1.3600e+03 & 5.2639e-03 & 1.3132e-01 & 8.0329e-02 & 5.9066e+01 \\
FiniteDiff+laplacian:0.1       & converged & 6.8260e-02 & 1.8000e+01 & 2.9600e+03 & 4.6332e-03 & 3.6945e-01 & 2.5795e-01 & 8.7145e+01 \\
\bottomrule
\end{tabular}%
}
\end{table}

\begin{table}[h]
\centering
\scriptsize
\caption{Nonlinear (High-noise) Data Physical Correctness, best Newton variant per method family}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccc}
\toprule
method & entropy\_violation\_\% & worst\_entropy\_violation & deriv\_violation\_\% & worst\_deriv\_violation \\
\midrule
Smooth+PCHIP+gradient:0.3      & 1.0000e+00 & 4.5721e-03 & 0.0000e+00 & 0.0000e+00 \\
RFF+gradient:0.3               & 1.2000e+00 & 1.6326e-02 & 0.0000e+00 & 0.0000e+00 \\
MLP                            & 2.0000e+00 & 1.0816e-02 & 0.0000e+00 & 0.0000e+00 \\
KISS-GP                        & 4.0000e-01 & 1.1288e-04 & 0.0000e+00 & 0.0000e+00 \\
FDIntegratedEpanechnikov+laplacian:0.01 & 4.0000e-01 & 3.2717e-04 & 0.0000e+00 & 0.0000e+00 \\
RBF                            & 2.6800e+01 & 2.2580e+00 & 2.9000e+01 & 8.2432e+00 \\
SavGol                         & 1.0000e+00 & 3.1677e-03 & 0.0000e+00 & 0.0000e+00 \\
PCHIP+gradient:0.3             & 1.0000e+00 & 1.1183e-02 & 0.0000e+00 & 0.0000e+00 \\
FDMatern52+laplacian:0.1       & 1.4000e+00 & 8.6212e-03 & 0.0000e+00 & 0.0000e+00 \\
FiniteDiff+laplacian:0.1       & 1.4000e+00 & 1.7192e-02 & 0.0000e+00 & 0.0000e+00 \\
\bottomrule
\end{tabular}%
}
\end{table}

## Nonlinear (Low-noise)

\begin{table}[h]
\centering
\scriptsize
\caption{Nonlinear (Low-noise) Data Accuracy, best Newton variant per method family}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
method & q RMSE & q/noise & $q_s$ RMSE & $q_s$/noise & $q_T$ RMSE & $q_T$/noise \\
\midrule
MLP                            & 2.4704e-01 & 6.3274e+00 & 5.5458e-01 & 6.0283e+00 & 7.3166e-01 & 3.9851e+00 \\
RFF                            & 5.2698e-02 & 1.1158e+00 & 8.9904e-02 & 9.7727e-01 & 2.5923e-01 & 1.4120e+00 \\
FDIntegratedEpanechnikov       & 3.9233e-01 & 5.5845e+00 & 5.1252e-01 & 5.5712e+00 & 2.3854e-01 & 1.2993e+00 \\
KISS-GP                        & 1.2158e-01 & 2.0877e+00 & 3.2440e-01 & 3.5262e+00 & 3.0935e-01 & 1.6849e+00 \\
SavGol                         & 4.9764e-02 & 1.0658e+00 & 5.1135e-02 & 5.5584e-01 & 7.4313e-02 & 4.0475e-01 \\
Smooth+PCHIP+gradient:0.3      & -- & -- & -- & -- & -- & -- \\
FDMatern52+gradient:0.3        & 2.4488e-01 & 3.6514e+00 & 3.0246e-01 & 3.2878e+00 & 2.0505e-01 & 1.1169e+00 \\
PCHIP+laplacian:0.1            & -- & -- & -- & -- & -- & -- \\
FiniteDiff+gradient:0.3        & 2.3721e-01 & 3.9427e+00 & 3.0742e-01 & 3.3416e+00 & 3.2895e-01 & 1.7917e+00 \\
RBF+gradient:0.3               & 3.3126e+00 & 1.0964e+02 & 5.4327e+00 & 5.9054e+01 & 1.0371e+01 & 5.6488e+01 \\
\bottomrule
\end{tabular}%
}
\end{table}

\begin{table}[h]
\centering
\scriptsize
\caption{Nonlinear (Low-noise) Newton Solve Results, best Newton variant per method family}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
method & status & build\_s & newton\_steps & flux\_calls & rel\_solution\_err & solve\_total\_s & flux\_eval\_s & avg\_flux\_eval\_us \\
\midrule
MLP                            & converged & 4.1126e+00 & 3.0000e+00 & 5.6000e+02 & 5.1526e-02 & 5.4928e+00 & 5.4510e+00 & 9.7339e+03 \\
RFF                            & converged & 2.2813e-01 & 4.0000e+00 & 7.2000e+02 & 3.6017e-03 & 2.8674e-01 & 2.2632e-01 & 3.1433e+02 \\
FDIntegratedEpanechnikov       & converged & 7.6649e-02 & 4.0000e+00 & 7.2000e+02 & 4.0086e-03 & 6.9432e-02 & 4.2625e-02 & 5.9201e+01 \\
KISS-GP                        & converged & 3.6565e-01 & 4.0000e+00 & 7.2000e+02 & 1.2076e-02 & 2.1999e+00 & 2.1474e+00 & 2.9825e+03 \\
SavGol                         & converged & 6.5894e-02 & 5.0000e+00 & 8.8000e+02 & 2.8787e-03 & 1.5709e-01 & 1.2252e-01 & 1.3923e+02 \\
Smooth+PCHIP+gradient:0.3      & converged & 7.0046e-02 & 5.0000e+00 & 8.8000e+02 & 3.6040e-03 & 3.3282e-01 & 2.9510e-01 & 3.3534e+02 \\
FDMatern52+gradient:0.3        & converged & 7.4550e-02 & 7.0000e+00 & 1.2000e+03 & 3.2397e-03 & 1.1551e-01 & 7.1037e-02 & 5.9197e+01 \\
PCHIP+laplacian:0.1            & converged & 6.9713e-02 & 7.0000e+00 & 1.2800e+03 & 5.9857e-03 & 4.8421e-01 & 4.2969e-01 & 3.3570e+02 \\
FiniteDiff+gradient:0.3        & converged & 6.9151e-02 & 1.1000e+01 & 1.8400e+03 & 5.1395e-03 & 2.3401e-01 & 1.6301e-01 & 8.8595e+01 \\
RBF+gradient:0.3               & not converged & 6.9393e-02 & -- & 1.1360e+04 & -- & 6.5894e-01 & 2.4994e-01 & 2.2002e+01 \\
\bottomrule
\end{tabular}%
}
\end{table}

\begin{table}[h]
\centering
\scriptsize
\caption{Nonlinear (Low-noise) Data Physical Correctness, best Newton variant per method family}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccc}
\toprule
method & entropy\_violation\_\% & worst\_entropy\_violation & deriv\_violation\_\% & worst\_deriv\_violation \\
\midrule
MLP                            & 1.2000e+00 & 5.5856e-03 & 0.0000e+00 & 0.0000e+00 \\
RFF                            & 2.0000e-01 & 5.4179e-05 & 0.0000e+00 & 0.0000e+00 \\
FDIntegratedEpanechnikov       & 6.0000e-01 & 3.0341e-04 & 0.0000e+00 & 0.0000e+00 \\
KISS-GP                        & 0.0000e+00 & 0.0000e+00 & 0.0000e+00 & 0.0000e+00 \\
SavGol                         & 2.0000e-01 & 3.7291e-05 & 0.0000e+00 & 0.0000e+00 \\
Smooth+PCHIP+gradient:0.3      & 8.0000e-01 & 6.4672e-04 & 0.0000e+00 & 0.0000e+00 \\
FDMatern52+gradient:0.3        & 8.0000e-01 & 6.5595e-04 & 0.0000e+00 & 0.0000e+00 \\
PCHIP+laplacian:0.1            & 1.4000e+00 & 1.4560e-02 & 0.0000e+00 & 0.0000e+00 \\
FiniteDiff+gradient:0.3        & 1.0000e+00 & 1.1131e-02 & 0.0000e+00 & 0.0000e+00 \\
RBF+gradient:0.3               & 2.9400e+01 & 1.7753e+01 & 3.5800e+01 & 1.1044e+01 \\
\bottomrule
\end{tabular}%
}
\end{table}

## Coverage Check

| CSV | requested families shown | not-converged rows | missing requested families |
|---|---:|---:|---|
| `linear_medium_noise.csv` | 10/10 | 1 | none |
| `nonlinear_high_noise.csv` | 10/10 | 0 | none |
| `nonlinear_low_noise.csv` | 10/10 | 1 | none |
