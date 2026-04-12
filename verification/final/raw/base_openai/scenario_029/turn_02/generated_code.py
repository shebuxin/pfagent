# required_dependencies: andes
import andes
import numpy as np

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add new PQ load at bus 4
new_load_idx = 'PQ_VERIFY_029_A'
new_load_bus = 4
p0 = 0.01
q0 = 0.006

ssa.PQ.add(new_load_idx, bus=new_load_bus, p0=p0, q0=q0)

# Rerun the power flow
andes.run_power_flow(ssa)

# Threshold for voltage
threshold = 0.950

# Get voltage of all buses
bus_voltages = ssa.Bus.v.v
bus_ids = ssa.Bus.idx.v

# Find buses below the threshold
selected_bus_ids = bus_ids[bus_voltages < threshold]
selected_bus_ids = selected_bus_ids.tolist()

# Get minimum voltage bus information
min_voltage = np.min(bus_voltages)
min_bus_index = np.where(bus_voltages == min_voltage)[0][0]
min_bus = int(ssa.Bus.idx.v[min_bus_index])

# Prepare RESULT_JSON
RESULT_JSON = {
    "added_load_idx": new_load_idx,
    "added_load_bus": new_load_bus,
    "threshold": float(threshold),
    "selected_bus_ids": sorted(selected_bus_ids),
    "selected_count": len(selected_bus_ids),
    "min_bus": min_bus,
    "min_voltage": round(float(min_voltage), 6)
}

# Output the result
print("RESULT_JSON=", RESULT_JSON)