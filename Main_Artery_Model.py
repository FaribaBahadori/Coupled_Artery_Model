
# Check required Python packages
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.getcwd())

data_dir = os.path.join(BASE_DIR, "Data")
plots_dir = os.path.join(BASE_DIR, "Plots")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)


# =====================================================================
# Run CellML model and obtain AM/AMp data
# =====================================================================
dt=1
sim_time=5000  #for steady state 100000s
pre_time=0  #s

from Functions.model_utils_artery import ArteryCellMLRunner


# CellML model
cellml_path = os.path.join(
    BASE_DIR,
    "CellML_Model",
    "Main_Coupled_Model1.cellml"
)


# Date/time for this artery-model run
run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

excel_file = os.path.join(
    data_dir,
    f"AM_AMp_Artery_Input_{run_timestamp}.xlsx"
)


# Create CellML runner
cellml_runner = ArteryCellMLRunner(
    model_path=cellml_path,
    dt=dt,
    sim_time=sim_time,
    pre_time=pre_time
)


# Run CellML and save AM/AMp data
cellml_runner.save_AM_AMp(excel_file)


# =====================================================================
# Read AM/AMp data
# =====================================================================

data = pd.read_excel(excel_file)

# Time is the first column
Time = data.iloc[:, 0].to_numpy()

# The latest CellML run is stored in the last two columns
AM = data.iloc[:, -2].to_numpy()
AMP = data.iloc[:, -1].to_numpy()


# =====================================================================
# Plot AM and AMp dynamics
# =====================================================================

plot_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

plt.figure(figsize=(10, 6))

plt.plot(Time, AM, label="AM", linewidth=3)
plt.plot(Time, AMP, label="AMp", linewidth=3)

plt.xlabel("Time (s)", fontweight="bold", fontsize=14)
plt.ylabel("Cross-bridge state", fontweight="bold", fontsize=14)
plt.title("AM and AMp Dynamics", fontweight="bold", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

AM_AMP_plot = os.path.join(
    plots_dir,
    f"AM_AMP_{plot_timestamp}.jpg"
)

plt.savefig(AM_AMP_plot, dpi=300, format="jpg")
plt.close()


# =====================================================================
# Define parameters
# =====================================================================

N = float(os.environ.get("N", "5000")) # cell ring numbers
kpp = float(os.environ.get("kpp", "0.1")) #uN
h = float(os.environ.get("h", "15")) #um
n = float(os.environ.get("n", "6")) #number of SMCs in the ring
L = float(os.environ.get("L", "10000")) #um

ls0 = float(os.environ.get("ls0", "30")) #um
l0 = float(os.environ.get("l0", "40")) #um

kx1 = float(os.environ.get("kx1", "12.5"))
kx2 = float(os.environ.get("kx2", "8.8"))
beta = float(os.environ.get("beta", "7.5"))
lopt = float(os.environ.get("lopt", "150")) #um
alpha_pp = float(os.environ.get("alpha_pp", "0.002"))

alpha_s = float(os.environ.get("alpha_s", "4.5"))
vx = float(os.environ.get("vx", "5000")) #um/s  # 5 um/ms
fAMp = float(os.environ.get("fAMp", "0.0013")) #uN.ms/um
fAM = float(os.environ.get("fAM", "0.0855")) #uN.ms/um
mu_s = float(os.environ.get("mu_s", "0.00001")) #uN.ms/um
ks = float(os.environ.get("ks", "0.2")) #uN

# Numerical safeguard # avoiding division by zero
epsilon1 = 1e-15


# ---------------------------------------------------------------------
# Pressure
# I Converted pressure unit from mmHg to µN/µm²
# ---------------------------------------------------------------------

# Pressure input is supplied in mmHg and converted to µN/µm²
PPi_input = float(os.environ.get("PPi", "30")) #mmHg
PPi = PPi_input * 133.322387415e-6 # uN/um²=10^6 pa #


# ---------------------------------------------------------------------
# Initial conditions and SMC mechanical parameters
# I converted parameter units, unit time from ms to s
# ---------------------------------------------------------------------

la00 = 89.60 #um
ls00 = 30 #um



# ---------------------------------------------------------------------
# Geometric parameter
# ---------------------------------------------------------------------

A = L * h


# =====================================================================
# Initialize arrays
# =====================================================================

LC = np.zeros(len(Time))
LA = np.zeros(len(Time))
LS = np.zeros(len(Time))


# =====================================================================
# Time integration
# =====================================================================

for j in range(len(Time) - 1):
    
    # Define the function in Eq 14 from /yang 2003b -vessel model
    def Eq14(lc_val):
        """Equation 14 in Python form."""

        return ((N * kpp) / A) * (
            np.exp(alpha_pp * (lc_val - l0) / l0) - 1
        ) \
        + (N / A) * (
     #       kx1 * AMP[j] + kx2 * AM[j]
             kx1 * 0 + kx2 * 0
        ) * (
            lc_val - la00 - ls00
        ) * np.exp(
            -beta * ((la00 - lopt) / lopt) ** 2
        ) \
        - (PPi / 2) * (
            ((n * lc_val) / (np.pi * h)) - 1
        )
    # Initial estimate for lc
    lc00 = la00 + ls00  

    # Solve the nonlinear algebraic equation for lc
    lc_solution = fsolve(Eq14, lc00)[0]

    # Current CellML contractile states
    lc = lc_solution
    ###Amp = AMP[j]
    ###Am = AM[j]
    #Test_AM=0_AMp=0
    Amp=0
    Am=0


    # Solve the ODE system for la and ls
    sol = solve_ivp(
        lambda t, y: ODE_La_Ls(
            t,
            y,
            lc,
            Amp,
            Am,
            ls0,
            kx1,
            kx2,
            beta,
            lopt,
            alpha_s,
            vx,
            fAMp,
            fAM,
            mu_s,
            ks,
            epsilon1
        ),
        [Time[j], Time[j + 1]],
        [la00, ls00],
        method='LSODA',
        rtol=1e-7,
        atol=1e-9
    )


    # Update mechanical state
    la00 = sol.y[0, -1]
    ls00 = sol.y[1, -1]
    lc00 =lc_solution


    # Store results
    LA[j] = la00
    LS[j] = ls00
    LC[j] = lc_solution


# =====================================================================
# Calculate final equilibrium state
# =====================================================================

def Eq14_final(lc_val):

    return ((N * kpp) / A) * (
        np.exp(alpha_pp * (lc_val - l0) / l0) - 1
    ) \
    + (N / A) * (
        kx1 * AMP[-1] + kx2 * AM[-1]
    ) * (
        lc_val - la00 - ls00
    ) * np.exp(
        -beta * ((la00 - lopt) / lopt) ** 2
    ) \
    - (PPi / 2) * (
        ((n * lc_val) / (np.pi * h)) - 1
    )


# Solve final equilibrium state
lc_final = fsolve(
    Eq14_final,
    la00 + ls00
)[0]


# Store final values
LC[-1] = lc_final
LA[-1] = la00
LS[-1] = ls00


# =====================================================================
# Calculate derived variables
# =====================================================================

LX = LC - LA - LS

Ri = 0.5 * (
    ((n * LC) / np.pi) - h
)

Di = 2 * Ri
Ro = Ri+h

# Calculate passive and active SMC forces
FP = kpp * (
    np.exp(alpha_pp * (LC - l0) / l0) - 1
)

FA = (
    (kx1 * AMP + kx2 * AM)
    * LX
    * np.exp(-beta * ((LA - lopt) / lopt) ** 2)
)

#sigma_t =PPi* Ri/h

sigma_a = (N/A)*FA
sigma_P = (N/A)*FP
sigma_t=sigma_a + sigma_P

# Plot passive and active SMC forces
force_plot_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

plt.figure(figsize=(10, 6))


plt.plot(Time, FA, label="Active force, Fa", linewidth=3)
plt.plot(Time, FP, label="Passive force, Fp", linewidth=3)

plt.xlabel("Time (s)", fontweight="bold", fontsize=14)
plt.ylabel("Force (μN)", fontweight="bold", fontsize=14)
plt.title("Passive and Active SMC Forces", fontweight="bold", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

force_plot_file = os.path.join(
    plots_dir,
    f"Fp_Fa_{force_plot_timestamp}.jpg"
)

plt.savefig(force_plot_file, dpi=300, format="jpg")
plt.close()

#====================================================================
# Plot passive and active SMC stress
stress_plot_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

plt.figure(figsize=(10, 6))

plt.plot(Time, sigma_a, label="Active stress, sigma_a", linewidth=3)
plt.plot(Time, sigma_P, label="Passive stress, sigma_P", linewidth=3)

plt.xlabel("Time (s)", fontweight="bold", fontsize=14)
plt.ylabel("Stress (μN/μm²)", fontweight="bold", fontsize=14)
plt.title("Passive and Active SMC Stresses", fontweight="bold", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

stress_plot_file = os.path.join(
    plots_dir,
    f"sigma_a_sigma_P_{stress_plot_timestamp}.jpg"
)

plt.savefig(stress_plot_file, dpi=300, format="jpg")
plt.close()

#====================================================================
# Plot passive and active SMC stress
stresst_plot_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

plt.figure(figsize=(10, 6))


plt.plot(Time, sigma_t, label="Total circumferential stress, sigma_t= sigma_a + sigma_P", linewidth=3)

plt.xlabel("Time (s)", fontweight="bold", fontsize=14)
plt.ylabel("Stress (μN/μm²)", fontweight="bold", fontsize=14)
plt.title("circumferential Stress", fontweight="bold", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

stresst_plot_file = os.path.join(
    plots_dir,
    f"sigma_t_{stresst_plot_timestamp}.jpg"
)

plt.savefig(stresst_plot_file, dpi=300, format="jpg")
plt.close()

# =====================================================================
# Save artery model results
# =====================================================================

artery_results = pd.DataFrame({
    "Time": Time,
    "LC": LC,
    "LA": LA,
    "LS": LS,
    "LX": LX,
    "Ri": Ri,
    "sigma_t": sigma_t,
    "F_P": FP,
    "F_a": FA
})


# Date/time for this artery-model run
run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# Output filename
artery_output_file = os.path.join(
    data_dir,
    f"Artery_Model_Output_{run_timestamp}.xlsx"
)


# Save results
artery_results.to_excel(
    artery_output_file,
    index=False
)


# =====================================================================
# Plotting
# =====================================================================

plt.figure(figsize=(10, 6))

plt.plot(
    Time,
    LC,
    label='LC',
    linewidth=3,
    color='blue'
)

plt.plot(
    Time,
    LA,
    label='LA',
    linewidth=3,
    color='black'
)

plt.plot(
    Time,
    LS,
    label='LS',
    linewidth=3,
    color='green'
)

plt.plot(
    Time,
    LX,
    label='LX',
    linewidth=3,
    color='yellow'
)

plt.plot(
    Time,
    Ri,
    label='Ri',
    linewidth=3,
    color='red'
)


plt.xlabel(
    'Time',
    fontweight='bold',
    fontsize=14
)

plt.ylabel(
    'LC and Ri',
    fontweight='bold',
    fontsize=14
)


plt.title(
    f'LC and Ri Over Time (PPi = {PPi_input:g} mmHg, Pre_time = {pre_time:g} s)',
    fontweight='bold',
    fontsize=14
)


plt.legend(
    loc='best',
    fontsize=12
)

plt.grid(True)
plt.tight_layout()


plot_file = os.path.join(
    plots_dir,
    f"Artery_Model_Plot_{run_timestamp}.png"
)


plt.savefig(
    plot_file,
    dpi=300
)

plt.close()
