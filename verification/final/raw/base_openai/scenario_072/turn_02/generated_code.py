# required_dependencies: numpy,andes
import numpy as np
import andes
import os

script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_072.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add new PQ load
new_load_idx = 'PQ_VERIFY_072_A'
new_load_bus = 6
p0 = 0.013
q0 = 0.008

# Add PQ load at bus 6
ssa.PQ.add(idx=new_load_idx, bus=new_load_bus, p0=p0, q0=q0)

# Run power flow
andes.run_power_flow(ssa)

# Get bus voltages and find buses below the threshold
threshold = 0.950
bus_voltages = ssa.Bus.vmag.v
bus_ids = ssa.Bus.idx.v

# Identify buses below the threshold
selected_bus_indices = np.where(bus_voltages < threshold)[0]
selected_bus_ids = bus_ids[selected_bus_indices].astype(int).tolist()

# Find the minimum voltage bus
min_bus_idx = np.argmin(bus_voltages)
min_bus = int(bus_ids[min_bus_idx])
min_voltage = round(bus_voltages[min_bus_idx], 6)

# Prepare result
RESULT_JSON = {
    "added_load_idx": new_load_idx,
    "added_load_bus": new_load_bus,
    "threshold": threshold,
    "selected_bus_ids": sorted(selected_bus_ids),
    "selected_count": len(selected_bus_ids),
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

print("RESULT_JSON=", RESULT_JSON)