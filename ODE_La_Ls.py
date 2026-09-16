import numpy as np

def ODE_La_Ls(t, y, lc, Amp, Am, ls0, kx1, kx2, beta, lopt):
    """
    Equivalent to MATLAB's ODE_La_Ls.m
    la = y[0], ls = y[1]
    """
    dy = np.zeros(2)

    # Parameters (same as in MATLAB)
    alpha_s = 4.5
    vx = 5000 #5 um/ms
    fAMp = 0.0013  #1.3 uN.ms/um
    fAM = 0.0855  #85.5 uN.ms/um where u=micro
    mu_s = 0.00001
    ks = 0.2   #uN
    epsilon = 1e-15

    # Differential equations
    dy[0] = ((kx1 * Amp + kx2 * Am) * (lc - y[0] - y[1]) - fAMp * Amp * vx) / (fAM * Am + fAMp * Amp + epsilon)
    dy[1] = ((kx1 * Amp + kx2 * Am) * (lc - y[0] - y[1])
             * np.exp(-beta * ((y[0] - lopt) / lopt) ** 2)
             - ks * (np.exp(alpha_s * (y[1] - ls0) / ls0) - 1)) / mu_s

    return dy
