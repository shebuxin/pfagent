# required_dependencies: andes,json
import andes
import json

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.015
pv_bus_idx = ssa.PV.idx.v[0]
ssa.PV.vset.v[pv_bus_idx] = 1.015

# Rerun power flow
result = andes.run_power_flow(ssa)

# Get the affected PV bus voltage
pv_voltage = round(result.V.mag.v[pv_bus_idx], 6)

# Count how many buses are above 1.020 p.u.
selected_count = (result.V.mag.v > 1.020).sum()

# Prepare result JSON
result_json = {
    "pv_bus": int(pv_bus_idx),
    "pv_setpoint": float(1.015),
    "pv_voltage": float(pv_voltage),
    "selected_count": int(selected_count)
}

# Print result
print("RESULT_JSON=" + json.dumps(result_json))