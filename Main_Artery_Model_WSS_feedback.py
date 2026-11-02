
# Check required Python packages
from xml.etree.ElementTree import PI

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
from scipy.optimize import brentq, minimize_scalar

import subprocess
try:
    import openpyxl
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "openpyxl"
    ])
import openpyxl


#start------- File paths -------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.getcwd())

data_dir = os.path.join(BASE_DIR, "Data")
plots_dir = os.path.join(BASE_DIR, "Plots")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

#start------- SMC voltage -------

voltage_mode = "step"  # "constant" or "step"

V0 = -60.0 #mV
V1 = -30.0 #mV

V_step_start = 200 #266.0 #seconds
V_step_end = 600 #2066.0  #seconds

def smc_voltage(t):
    if voltage_mode == "constant":
        return V0

    if V_step_start <= t <= V_step_end:
        return V1

    return V0


#start------- Simulation settings -------
from Functions.model_utils_artery import ArteryCellMLRunner

wss_mode = "feedback"       # "constant" or "feedback"
tau_ref = 20.0              # dyn/cm²; replaces CellML Environment/tau_c
coupling_dt = 1           # s; start at 0.1, then test 0.5 or 1.0
sim_time = 1000           # s
pre_time = 5.0              # feedback implementation requires zero pre-time

if wss_mode not in ("constant", "feedback"):
    raise ValueError("wss_mode must be 'constant' or 'feedback'")
if coupling_dt <= 0 or sim_time <= 0:
    raise ValueError("coupling_dt and sim_time must be positive")
if pre_time < 0:
    raise ValueError("pre_time must be non-negative")
if tau_ref < 0:
    raise ValueError("tau_ref must be non-negative")

n_intervals = int(round(sim_time / coupling_dt))
if not np.isclose(n_intervals * coupling_dt, sim_time, rtol=0, atol=1e-9):
    raise ValueError("sim_time must be an integer multiple of coupling_dt")

Time = np.linspace(0, sim_time, n_intervals + 1)

V_SMC = np.array([smc_voltage(t) for t in Time])

cellml_path = os.path.join(BASE_DIR, "CellML_Model", "Main_Coupled_Model1.cellml")
run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
excel_file = os.path.join(data_dir, f"AM_AMp_Artery_Input_{run_timestamp}.xlsx")
cellml_runner = ArteryCellMLRunner(
    model_path=cellml_path, dt=coupling_dt,
    sim_time=sim_time, pre_time=0
)
helper = cellml_runner.sim_object

# SMC membrane voltage parameters--------------------------------

voltage_params = {
    "SMC_Par/V0": V0,
    "SMC_Par/V1": V0 if voltage_mode == "constant" else V1,
    "SMC_Par/t1_KCL": V_step_start,
    "SMC_Par/t2_KCL": V_step_end,
}

helper.set_param_vals(
    list(voltage_params.keys()),
    list(voltage_params.values())
)
# WSS input
helper.set_param_vals(["Environment/tau_c"], [tau_ref])

# Pre-simulation
if pre_time > 0:
    helper.update_times(
        coupling_dt, 0.0, pre_time, 0.0
    )

    if not helper.run():
        raise RuntimeError("CellML pre-simulation failed")

#start------- Artery parameters -------

N = float(os.environ.get("N", "5000")) # cell ring numbers
kpp = float(os.environ.get("kpp", "0.1")) #uN
h = float(os.environ.get("h", "15")) #um
n = float(os.environ.get("n", "6")) #number of SMCs in the ring
L = float(os.environ.get("L", "10000")) #um

ls0 = float(os.environ.get("ls0", "37.05")) #um  #30
l0 = float(os.environ.get("l0", "40")) #um

kx1 = float(os.environ.get("kx1", "12.5"))
kx2 = float(os.environ.get("kx2", "8.8"))
beta = float(os.environ.get("beta", "7.5"))
lopt = float(os.environ.get("lopt", "150")) #um
alpha_pp = float(os.environ.get("alpha_pp", "1.2")) #0.2

alpha_s = float(os.environ.get("alpha_s", "4.5"))
vx = float(os.environ.get("vx", "5000")) #um/s  # 5 um/ms
fAMp = float(os.environ.get("fAMp", "0.0013")) #uN.ms/um
fAM = float(os.environ.get("fAM", "0.0855")) #uN.ms/um
mu_s = float(os.environ.get("mu_s", "0.00001")) #0.01 uN.ms/um
ks = float(os.environ.get("ks", "0.2")) #uN

epsilon1 = 1e-15


# Pressure
#start------- Pressure input -------

pressure_mode = "constant"  # "constant" or "step"
P0i = 30               # Constant pressure (mmHg)

def pressure(t):
    if pressure_mode == "constant":
        return P0i

    if t < 540:
        return 10
    elif t < 720:
        return 60
    elif t < 930:
        return 10
    elif t < 1110:
        return 60
    elif t < 1320:
        return 10
    elif t < 1500:
        return 60
    else:
        return 10

# Pressure input is supplied in mmHg and converted to µN/µm²
#PPi_input = float(os.environ.get("PPi", "60")) #mmHg
#PPi = PPi_input * 133.322387415e-6 # uN/um²=10^6 pa #


#start------- Initial conditions -------

la00 = 89.60 #um
ls00 = 30 #um 



#start------- Geometry -------

A = L * h


#start------- Coupling and mechanical equilibrium -------
LC = np.empty(len(Time))
LA = np.empty(len(Time))
LS = np.empty(len(Time))
PI = np.empty(len(Time))
AM = np.empty(len(Time))
AMP = np.empty(len(Time))
WSS = np.empty(len(Time))

lower = 2.5 * np.pi
upper = 2000.0

def solve_lc(la, ls, am, amp, pressure_mmhg, time_s):
    """Eq. 14: largest admissible root."""
    p = pressure_mmhg * 133.322387415e-6
    def eq14(lc_val):
        return ((N * kpp) / A) * (np.exp(alpha_pp * (lc_val - l0) / l0) - 1) \
            + (N / A) * (kx1 * amp + kx2 * am) * (lc_val - la - ls) \
            * np.exp(-beta * ((la - lopt) / lopt) ** 2) \
            - (p / 2) * (((n * lc_val) / (np.pi * h)) - 1)

    minimum = minimize_scalar(eq14, bounds=(lower, upper), method="bounded").x
    if eq14(minimum) <= 0 and eq14(upper) >= 0:
        return brentq(eq14, minimum, upper)
    if eq14(lower) * eq14(minimum) < 0:
        return brentq(eq14, lower, minimum)
    raise ValueError(f"No admissible LC root at t={time_s:.3f} s")

def diameter_from_lc(lc):
    return n * lc / np.pi - h

# Initial cross-bridge states
initial_states = helper.data.states()

AM[0] = float(initial_states["AM/AM"])

AMP[0] = 1.0 - (
    float(initial_states["M/M"])
    + float(initial_states["AM/AM"])
    + float(initial_states["Mp/Mp"])
)
LA[0] = la00
LS[0] = ls00
PI[0] = pressure(Time[0])
LC[0] = solve_lc(LA[0], LS[0], AM[0], AMP[0], PI[0], Time[0])
D_ref = diameter_from_lc(LC[0])
if D_ref <= 0:
    raise ValueError(f"Non-positive initial diameter: {D_ref}")
WSS[0] = tau_ref
print(f"Reference diameter: {D_ref:.6g} µm; reference WSS: {tau_ref:g} dyn/cm²")

# Constant WSS
if wss_mode == "constant":
    if not helper.run():
        raise RuntimeError("Continuous CellML run failed")
    AM[:] = np.asarray(helper.get_results(["AM/AM"])[0][0], dtype=float)
    AMP[:] = np.asarray(helper.get_results(["AMp/AMp"])[0][0], dtype=float)
    if len(AM) != len(Time) or len(AMP) != len(Time):
        raise RuntimeError("CellML output length does not match Time")
    LC[0] = solve_lc(LA[0], LS[0], AM[0], AMP[0], PI[0], Time[0])
    D_ref = diameter_from_lc(LC[0])
    if D_ref <= 0:
        raise ValueError("Non-positive reference diameter")

for j in range(len(Time) - 1):
    if wss_mode == "feedback":
        helper.set_param_vals(["Environment/tau_c"], [float(WSS[j])])
        helper.update_times(coupling_dt, float(Time[j]), coupling_dt, 0.0)
        if not helper.run():
            raise RuntimeError(f"CellML failed at t={Time[j]:.3f} s")
        AM[j + 1] = float(helper.get_results(["AM/AM"])[0][0][-1])
        AMP[j + 1] = float(helper.get_results(["AMp/AMp"])[0][0][-1])

    sol = solve_ivp(
        lambda t, y: ODE_La_Ls(
            t, y, LC[j], AMP[j], AM[j], ls0, kx1, kx2,
            beta, lopt, alpha_s, vx, fAMp, fAM, mu_s, ks, epsilon1
        ),
        (float(Time[j]), float(Time[j + 1])),
        [LA[j], LS[j]],
        method="LSODA", rtol=1e-7, atol=1e-9
    )
    if not sol.success:
        raise RuntimeError(f"Mechanical integration failed at t={Time[j]:.3f}: {sol.message}")
    LA[j + 1], LS[j + 1] = sol.y[:, -1]
    PI[j + 1] = pressure(Time[j + 1])
    LC[j + 1] = solve_lc(
        LA[j + 1], LS[j + 1], AM[j + 1], AMP[j + 1],
        PI[j + 1], Time[j + 1]
    )
    di_next = diameter_from_lc(LC[j + 1])
    if di_next <= 0:
        raise ValueError(f"Non-positive diameter at t={Time[j + 1]:.3f} s")
    WSS[j + 1] = (
        tau_ref * (D_ref / di_next) ** 3
        if wss_mode == "feedback" else tau_ref
    )

run_info = pd.DataFrame({
    "Run date": [datetime.datetime.now().strftime("%Y-%m-%d")],
    "Run time": [datetime.datetime.now().strftime("%H:%M:%S")],
    "dt (s)": [coupling_dt],
    "Simulation time (s)": [sim_time],
    "CellML model": [cellml_path],
    "WSS mode": [wss_mode],
    "Reference WSS (dyn/cm²)": [tau_ref],
    "Reference diameter (µm)": [D_ref]
})
with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
    pd.DataFrame({"Time": Time, "AM_Run1": AM, "AMp_Run1": AMP}).to_excel(
        writer, sheet_name="AM_AMp_Data", index=False
    )
    run_info.to_excel(writer, sheet_name="Run_Information", index=False)
print("AM/AMp data:", excel_file)

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
plt.savefig(os.path.join(plots_dir, f"AM_AMP_{plot_timestamp}.jpg"), dpi=300, format="jpg")
plt.close()

LX = LC - LA - LS

# Mechanical rates
dla = np.empty(len(Time))
dls = np.empty(len(Time))
for j in range(len(Time)):
    rate = ODE_La_Ls(
        float(Time[j]), [LA[j], LS[j]], LC[j], AMP[j], AM[j],
        ls0, kx1, kx2, beta, lopt, alpha_s, vx,
        fAMp, fAM, mu_s, ks, epsilon1
    )
    dla[j], dls[j] = rate

Amp = AMP[-1]  # retained only for compatibility with original downstream variables
Am = AM[-1]
lc = LC[-1]
F_x = (kx1 * AMP + kx2 * AM) * LX * np.exp(-beta * ((LA - lopt) / lopt) ** 2)
F_a = (fAMp * AMP * (vx + dla) + fAM * AM * dla) * np.exp(-beta * ((LA - lopt) / lopt) ** 2)
F_s = mu_s * dls + ks * (np.exp(alpha_s * (LS - ls0) / ls0) - 1)
PPi_input = PI[-1]

#start------- Derived variables -------

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

#start------- Save artery results -------

artery_results = pd.DataFrame({
    "Time": Time,
    "LC": LC,
    "LA": LA,
    "LS": LS,
    "LX": LX,
    "Ri": Ri,
    "Di": Di,
    "sigma_t": sigma_t,
    "sigma_a": sigma_a,
    "sigma_P": sigma_P,
    "dla": dla,
    "dls": dls,
    "WSS": WSS,
    "V_SMC": V_SMC,


})


run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


artery_output_file = os.path.join(
    data_dir,
    f"Artery_Model_Output_{run_timestamp}.xlsx"
)


artery_results.to_excel(
    artery_output_file,
    index=False
)

#start------- WSS plot -------
plt.figure(figsize=(10, 6))
plt.plot(Time, WSS, label="WSS", linewidth=3)
plt.xlabel("Time (s)", fontweight="bold", fontsize=14)
plt.ylabel("WSS (dyn/cm²)", fontweight="bold", fontsize=14)
plt.title("Wall Shear Stress Dynamics", fontweight="bold", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
wss_plot_file = os.path.join(plots_dir, f"WSS_{run_timestamp}.jpg")
plt.savefig(wss_plot_file, dpi=300, format="jpg")
plt.close()
print("Artery results:", artery_output_file)
print("WSS plot:", wss_plot_file)

#start------- Artery plot -------

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
