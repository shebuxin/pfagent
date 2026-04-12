# required_dependencies: pandas, numpy
import os
import andes
import numpy as np

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_053.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target
first_pv_idx = ssa.PV.idx.v[0]
ssa.PV.volt_target.set(first_pv_idx, 1.015)

# Rerun power flow
andes.run(ssa)

# Get the affected PV bus voltage
pv_bus = int(ssa.PV.bus.v[first_pv_idx])
pv_setpoint = round(1.015, 6)
pv_voltage = round(float(ssa.V.v[pv_bus]), 6)

# Count how many buses are above 1.040 p.u.
above_threshold_count = np.sum(ssa.V.v > 1.040)

# Create the result JSON
RESULT_JSON = {
    "pv_bus": pv_bus,
    "pv_setpoint": pv_setpoint,
    "pv_voltage": pv_voltage,
    "selected_count": int(above_threshold_count)
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")