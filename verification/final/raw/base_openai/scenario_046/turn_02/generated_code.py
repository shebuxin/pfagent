# required_dependencies: numpy,andes
import os
import numpy as np
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_046.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 20
bus_20_idx = np.where(ssa.Bus.idx.v == 20)[0]
if bus_20_idx.size == 0:
    raise ValueError("Bus 20 not found in the case.")
    
ssa.PQ.add("PQ_VERIFY_046_A", p0=0.019, q0=0.012, bus=20)

# Setup the modified case
ssa.setup()

# Run power flow again
andes.power_flow(ssa)

# Get buses with voltages below 0.980 p.u.
threshold = 0.980
low_voltage_buses = np.where(ssa.Bus.v < threshold)[0]
selected_bus_ids = ssa.Bus.idx.v[low_voltage_buses].tolist()

# Find the minimum bus voltage and its index
min_voltage_idx = np.argmin(ssa.Bus.v)
min_bus = int(ssa.Bus.idx.v[min_voltage_idx])
min_voltage = round(ssa.Bus.v[min_voltage_idx], 6)

# Prepare the result JSON
RESULT_JSON = {
    "added_load_idx": "PQ_VERIFY_046_A",
    "added_load_bus": 20,
    "threshold": threshold,
    "selected_bus_ids": sorted(selected_bus_ids),
    "selected_count": len(selected_bus_ids),
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

# Print result
print(f"RESULT_JSON={RESULT_JSON}")