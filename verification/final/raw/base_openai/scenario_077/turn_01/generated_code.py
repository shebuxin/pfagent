# required_dependencies: pandas
import pandas as pd
import os
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_077.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow analysis
andes.run_power_flow(ssa)

# Count buses above 1.010 p.u.
threshold = 1.010
selected_bus_ids = ssa.Bus.idx.v[ssa.Bus.v.v > threshold].tolist()
selected_count = len(selected_bus_ids)

# Find the two lowest-voltage buses
lowest_indices = ssa.Bus.idx.v[np.argsort(ssa.Bus.v.v)][:2]
lowest_bus_ids = lowest_indices.tolist()
lowest_voltages = [round(ssa.Bus.v.v[i], 6) for i in lowest_indices]

# Form the result
RESULT_JSON = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

print(f"RESULT_JSON={RESULT_JSON}")