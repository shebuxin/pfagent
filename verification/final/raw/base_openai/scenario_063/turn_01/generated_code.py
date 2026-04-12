# required_dependencies: pandas
import pandas as pd
import andes

# Load the built-in Kundur full case
case = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Define threshold
threshold = 1.010

# Count buses above the threshold
above_threshold = ssa.Bus.voltage.v > threshold
selected_bus_ids = ssa.Bus.idx.v[above_threshold].tolist()
selected_count = len(selected_bus_ids)

# Find the two lowest-voltage buses
voltages = ssa.Bus.voltage.v
lowest_bus_ids = ssa.Bus.idx.v[voltages.argsort()][:2].tolist()
lowest_voltages = [round(voltages[bus_id], 6) for bus_id in lowest_bus_ids]

# Constructing the final JSON-like output
RESULT_JSON = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

print(f"RESULT_JSON={RESULT_JSON}")