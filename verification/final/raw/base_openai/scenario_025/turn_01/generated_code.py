# required_dependencies: pandas, numpy, andes
import pandas as pd
import numpy as np
import andes

# Load the case
case = "verify_ieee14_025.xlsx"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow analysis
andes.run_powerflow(ssa)

# Get all bus voltages
voltages = ssa.Bus.v.v

# Find maximum and minimum voltage buses
max_voltage = np.max(voltages)
min_voltage = np.min(voltages)

max_bus = np.where(voltages == max_voltage)[0][0] + 1  # Convert index to bus number (1-based)
min_bus = np.where(voltages == min_voltage)[0][0] + 1  # Convert index to bus number (1-based)

# Prepare results
RESULT_JSON = {
    "max_bus": int(max_bus),
    "max_voltage": round(float(max_voltage), 6),
    "min_bus": int(min_bus),
    "min_voltage": round(float(min_voltage), 6)
}

# Print the result in the required format
print(f"RESULT_JSON={RESULT_JSON}")