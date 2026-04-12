# required_dependencies: andes
import andes
import os
import json

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_079.xlsx")
ssa = andes.load(case, setup=True, no_output=True)

# Set the first PV voltage target to 0.990
first_pv_idx = ssa.PV.idx.v[0]
ssa.PV.set('Vtarget', [0.990], idx=[first_pv_idx])

# Rerun power flow analysis
andes.run_power_flow(ssa)

# Get the updated voltage of the affected PV bus
pv_voltage = round(ssa.V.bus.v[ssa.PV.bus.v[first_pv_idx]], 6)

# Count how many buses have voltage above 0.990 p.u.
above_count = (ssa.V.bus.v > 0.990).sum()

# Prepare the JSON result
result_json = {
    "pv_bus": int(ssa.PV.bus.v[first_pv_idx]),
    "pv_setpoint": float(0.990),
    "pv_voltage": float(pv_voltage),
    "selected_count": int(above_count),
}

# Print the result as specified
print(f"RESULT_JSON={json.dumps(result_json)}")