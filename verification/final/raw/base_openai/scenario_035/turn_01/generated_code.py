# required_dependencies: andes
import andes

case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Count all buses above 1.040 p.u.
threshold = 1.040
selected_bus_ids = ssa.Bus.idx.v[ssa.Bus.v.v > threshold].tolist()
selected_count = len(selected_bus_ids)

# Find the two lowest-voltage buses
lowest_bus_indices = ssa.Bus.idx.v[np.argsort(ssa.Bus.v.v)][:2]
lowest_bus_ids = ssa.Bus.idx.v[lowest_bus_indices].tolist()
lowest_voltages = [round(ssa.Bus.v.v[i], 6) for i in lowest_bus_indices]

# Prepare the JSON output
RESULT_JSON = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

print("RESULT_JSON=", RESULT_JSON)