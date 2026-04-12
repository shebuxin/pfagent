# required_dependencies: pandas, numpy
import andes
import pandas as pd
import numpy as np

# Load the Kundur full case
case = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Count all buses above 1.000 p.u.
threshold = 1.000
above_threshold_buses = ssa.Bus.idx.v[ssa.Bus.voltage.v > threshold]
selected_bus_ids = above_threshold_buses.tolist()
selected_count = len(selected_bus_ids)

# Get the two lowest-voltage buses
lowest_voltages_indices = np.argsort(ssa.Bus.voltage.v)[:2]
lowest_bus_ids = ssa.Bus.idx.v[lowest_voltages_indices].tolist()
lowest_voltages = [round(ssa.Bus.voltage.v[idx], 6) for idx in lowest_voltages_indices]

# Prepare RESULT_JSON
RESULT_JSON = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

# Print result
print(f"RESULT_JSON={RESULT_JSON}")