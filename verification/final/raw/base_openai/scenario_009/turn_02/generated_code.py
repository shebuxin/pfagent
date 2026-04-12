# required_dependencies: andes
import andes
import numpy as np

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.010
first_pv_idx = ssa.PV.idx.v[0]
ssa.PV.Vtarget.set(first_pv_idx, 1.010)

# Rerun power flow
andes.run(ssa)

# Get the affected PV bus voltage and the voltage setpoint
pv_bus = int(ssa.PV.bus.v[first_pv_idx])  # Convert to 1-based index
pv_setpoint = 1.010
pv_voltage = round(float(ssa.Bus.Vm.v[pv_bus - 1]), 6)  # Accessing bus voltage, adjusting for 0-based index

# Count how many buses have voltage above 1.015 p.u.
selected_count = int(np.sum(ssa.Bus.Vm.v > 1.015))

# Prepare the JSON result
RESULT_JSON = {
    "pv_bus": pv_bus,
    "pv_setpoint": pv_setpoint,
    "pv_voltage": pv_voltage,
    "selected_count": selected_count
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")