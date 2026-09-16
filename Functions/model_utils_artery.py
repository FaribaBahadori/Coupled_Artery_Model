#file name:model_utlis_artery.py
import os #D# Handles file/folder paths.
import sys #D# Allows Python to find modules/scripts in that folder.
import numpy as np #D# Used for handling AM/AMp numerical arrays.
import pandas as pd #D# Used for reading/writing CSV/Excel data.
import datetime

# Find the OpenCOR helper used by my existing model_utils_artery.py
sys.path.append(os.path.join(os.path.dirname(__file__), "OpenCor_Py")) #D# Adds my OpenCor_Py folder to Python's search path.

#Imports my existing OpenCOR helper, which actually opens and runs the CellML model.
from Functions.OpenCor_Py.opencor_helper import SimulationHelper

#Creates a reusable object for my CellML --> AM/AMp workflow.
class ArteryCellMLRunner:
    
    #D# Initializes the runner with the path to the CellML model and simulation parameters.
    #Sets: CellML file location, time step = 1 s, simulation duration = 1300 s, and pre-time = 0
    def __init__(self, model_path, dt=1, sim_time=1300, pre_time=0):
        #D# Stores the settings mentioned in the previous line.
        self.model_path = model_path
        self.dt = dt
        self.sim_time = sim_time
        self.pre_time = pre_time
        # Use the same OpenCOR solver settings as the working calibration model
        solver_info = {'MaximumStep': 0.1,'MaximumNumberOfSteps': 5000}

        # Create the OpenCOR simulation object using my existing opencor_helper.py
        self.sim_object = SimulationHelper(model_path,dt, sim_time, solver_info={'MaximumNumberOfSteps': 100000,'MaximumStep': 0.0001}, pre_time=pre_time)
    
    def run_cellml(self):

        # Run the CellML model using its current/default parameter values
        success = self.sim_object.run()

        print("CellML run returned:", success)

        if not success: # Stops and reports an error if OpenCOR fails.
            raise RuntimeError("CellML simulation failed.")

        # Get simulation time points, adjusted for pre_time.
        time = self.sim_object.tSim - self.pre_time

        # Get AM and AMp variables from the CellML model
        AM = self.sim_object.get_results(["AM/AM"])
        AMp = self.sim_object.get_results(["AMp/AMp"])

        # Converts them into simple 1-D arrays.
        AM = np.squeeze(AM)
        AMp = np.squeeze(AMp)

        return time, AM, AMp

    def save_AM_AMp(self, excel_file):

        # Run CellML and get AM and AMp results.
        time, AM, AMp = self.run_cellml()

        # Record information about this run
        run_info = pd.DataFrame({
            "Run date": [datetime.datetime.now().strftime("%Y-%m-%d")],
            "Run time": [datetime.datetime.now().strftime("%H:%M:%S")],
            "dt (s)": [self.dt],
            "Simulation time (s)": [self.sim_time],
            "CellML model": [self.model_path]
        })

        # If the Excel file already exists, read the existing AM/AMp data
        if os.path.exists(excel_file):

            old_data = pd.read_excel(
                excel_file,
                sheet_name="AM_AMp_Data"
            )

            run_number = (old_data.shape[1] - 1) // 2 + 1

            new_data = pd.DataFrame({
                f"AM_Run{run_number}": AM,
                f"AMp_Run{run_number}": AMp
            })

            old_data = pd.concat(
                [old_data, new_data],
                axis=1
            )

        else:

            run_number = 1

            old_data = pd.DataFrame({
                "Time": time,
                "AM_Run1": AM,
                "AMp_Run1": AMp
            })

        # Write both sheets while keeping the workbook together
        with pd.ExcelWriter(
            excel_file,
            engine="openpyxl",
            mode="w"
        ) as writer:

            old_data.to_excel(
                writer,
                sheet_name="AM_AMp_Data",
                index=False
            )

            run_info.to_excel(
                writer,
                sheet_name="Run_Information",
                index=False
            )

        # Clear simulation
        self.sim_object.reset_and_clear()

        print(f"AM and AMp saved to: {excel_file}")