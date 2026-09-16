# check requiored Python packages
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import datetime
import os
import sys
import pandas as pd
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from ODE_La_Ls import ODE_La_Ls
import subprocess
try:
    import openpyxl
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "openpyxl"
    ])
import openpyxl

# ---------------------------------------------------------------------
# Set the working directory to wherever this Python file is located.
# ---------------------------------------------------------------------
#os.chdir(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.getcwd())

data_dir = os.path.join(BASE_DIR, "Data")
plots_dir = os.path.join(BASE_DIR, "Plots") 
# =====================================================================
# >>> NEW: Run CellML model and update AM/AMp Excel file
# =====================================================================

from Functions.model_utils_artery import ArteryCellMLRunner

# CellML model
cellml_path = os.path.join(
    BASE_DIR,
    "CellML_Model",
    "Main_Coupled_Model1.cellml"
)

# Fixed Excel file containing AM and AMp data
# Date/time for this artery-model run
run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
excel_file = os.path.join(data_dir, f"AM_AMp_Artery_Input_{run_timestamp}.xlsx")

# Create CellML runner
cellml_runner = ArteryCellMLRunner(
    model_path=cellml_path,
    dt=1,
    sim_time=1300,
    pre_time=0
)

# Run CellML and add the new AM/AMp data to Excel
cellml_runner.save_AM_AMp(excel_file)

# =====================================================================
# >>> END NEW
# =====================================================================


# ---------------------------------------------------------------------
# Equivalent of: clearvars; close all; clc;
# ---------------------------------------------------------------------

# Global parameters
lc = None
Amp = None
Am = None
ls0 = None
kx1 = None
kx2 = None
beta = None
lopt = None


# ---------------------------------------------------------------------
# Reading data from the Excel file
# ---------------------------------------------------------------------

# >>> CHANGED: Read the latest AM and AMp run from Excel
data = pd.read_excel(excel_file)

# Time is always the first column
Time = data.iloc[:, 0].to_numpy()

# The latest CellML run is always the last two columns
AM = data.iloc[:, -2].to_numpy()
AMP = data.iloc[:, -1].to_numpy()


# ---------------------------------------------------------------------
# Defining parameters
# ---------------------------------------------------------------------

N = 5000
kpp = 0.1  #uN
h = 15  # um = micro-metre
n = 6
L=10000  #10000 um = 10 mm
ls0 = 30  # um = micro-metre
l0 = 40  # um
kx1 = 12.5  #uN/um = micro-Newton per micro-metre
kx2 = 8.8  #uN/um = micro-Newton per micro-metre
beta = 7.5
lopt = 150  # um = micro-metre
alpha_pp = 0.0002
PPi = 0.0021 #

# Initial conditions
la00 = 89.60 #um
ls00 = 30 # um
alpha_s = 4.5
vx = 5000   #5 um/ms
fAMp = 0.0013  #1.3 uN.ms/um
fAM = 0.0855  #85.5 uN.ms/um where u=micro
mu_s = 0.00001 
ks = 0.2  #uN = micro-Newton
epsilon = 1e-15

A = L * h


# ---------------------------------------------------------------------
# Initialize arrays
# ---------------------------------------------------------------------

LC = np.zeros(len(Time))
LA = np.zeros(len(Time))
LS = np.zeros(len(Time))


# ---------------------------------------------------------------------
# Loop over time points
# ---------------------------------------------------------------------

for j in range(len(Time) - 1):

    def Eq14(lc_val):
        """Equation 14 in Python form"""

        return ((N * kpp) / A) * (
            np.exp(alpha_pp * (lc_val - l0) / l0) - 1
        ) \
        + (N / A) * (
            kx1 * AMP[j] + kx2 * AM[j]
        ) * (
            lc_val - la00 - ls00
        ) * np.exp(
            -beta * ((la00 - lopt) / lopt) ** 2
        ) \
        - (PPi / 2) * (
            ((n * lc_val) / (np.pi * h)) - 1
        )

    lc00 = la00 + ls00

    lc_solution = fsolve(Eq14, lc00)[0]

    # Update globals for ODE function
    lc = lc_solution
    Amp = AMP[j]
    Am = AM[j]

    # Solve ODE system for la and ls
    sol = solve_ivp(
        lambda t, y: ODE_La_Ls(
            t, y, lc, Amp, Am,
            ls0, kx1, kx2, beta, lopt
        ),
        [Time[j], Time[j+1]],
        [la00, ls00],
        method='LSODA',
        rtol=1e-7,
        atol=1e-9
    )

    la00 = sol.y[0, -1]
    ls00 = sol.y[1, -1]

    LA[j] = la00
    LS[j] = ls00
    LC[j] = lc_solution


# ---------------------------------------------------------------------
# >>> NEW: Save artery model results
# ---------------------------------------------------------------------

# Calculate LX and Ri
LX = LC - LA - LS
Ri = 0.5 * (((n * LC) / np.pi) - h)

# Create output dataframe
artery_results = pd.DataFrame({
    "Time": Time,
    "LC": LC,
    "LA": LA,
    "LS": LS,
    "LX": LX,
    "Ri": Ri
})

# Date/time for this artery-model run
run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Output filename
artery_output_file = os.path.join(data_dir, f"Artery_Model_Output_{run_timestamp}.xlsx")

# Save results
artery_results.to_excel(
    artery_output_file,
    index=False
)

print(f"Artery model results saved to: {artery_output_file}")

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(Time, LC, label='LC', linewidth=3, color='blue')
plt.plot(Time, LA, label='LA', linewidth=3, color='black')
plt.plot(Time, LS, label='LS', linewidth=3, color='green')
plt.plot(Time, LX, label='LX', linewidth=3, color='yellow')
plt.plot(Time, Ri, label='Ri', linewidth=3, color='red')

plt.xlabel('Time', fontweight='bold', fontsize=14)
plt.ylabel('LC and Ri', fontweight='bold', fontsize=14)

plt.title(
    f'LC and Ri Over Time (PPi = {PPi})',
    fontweight='bold',
    fontsize=14
)

plt.legend(loc='best', fontsize=12)
plt.grid(True)
plt.tight_layout()

plot_file = os.path.join(plots_dir,f"Artery_Model_Plot_{run_timestamp}.png")
plt.savefig(plot_file, dpi=300)

plt.close()
print(f"Artery model plot saved to: {plot_file}")