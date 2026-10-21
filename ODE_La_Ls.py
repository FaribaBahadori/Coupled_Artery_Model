import numpy as np

def ODE_La_Ls(
    t, y, lc, Amp, Am,
    ls0, kx1, kx2, beta, lopt,
    alpha_s, vx, fAMp, fAM, mu_s, ks, epsilon1
):
    """
    Equivalent to MATLAB's ODE_La_Ls.m
    la = y[0], ls = y[1]
    """
    dy = np.zeros(2)

    # Parameters (same as in MATLAB)

    # Differential equations
    dy[0] = ((kx1 * Amp + kx2 * Am) * (lc - y[0] - y[1]) - fAMp * Amp * vx) / (fAM * Am + fAMp * Amp + epsilon1)
    dy[1] = ((kx1 * Amp + kx2 * Am) * (lc - y[0] - y[1])
             * np.exp(-beta * ((y[0] - lopt) / lopt) ** 2)
             - ks * (np.exp(alpha_s * (y[1] - ls0) / ls0) - 1)) / mu_s

    return dy
