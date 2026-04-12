# required_dependencies: pandas
import andes
import os
import json

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_075.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Count all buses above 0.990 p.u.
threshold = 0.990
selected_bus_ids = ssa.Bus.idx.v[ssa.Bus.vmag.v > threshold].tolist()
selected_count = len(selected_bus_ids)

# Find the two lowest-voltage buses
lowest_bus_ids = ssa.Bus.idx.v[np.argsort(ssa.Bus.vmag.v)[:2]].tolist()
lowest_voltages = [round(v, 6) for v in np.sort(ssa.Bus.vmag.v)[:2]]

# Prepare result in JSON format
result_json = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

# Print the result
print(f"RESULT_JSON={json.dumps(result_json)}")