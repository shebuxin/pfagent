Here’s the corrected script using the valid case path for the Kundur full case. It will run the power flow and report the 4 lowest-voltage buses as specified.

```python
# required_dependencies: andes
import andes

# Load the Kundur full case
case_path = andes.get_case("kundur/kundur_full.json")
ssa = andes.load(case_path, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Retrieve voltage data
voltages = ssa.Bus.v

# Find the 4 lowest-voltage buses
lowest_voltage_indices = voltages.argsort()[:4]
selected_bus_ids = ssa.Bus.idx.v[lowest_voltage_indices].tolist()
selected_voltages = [round(voltages[idx], 6) for idx in lowest_voltage_indices]

# Prepare the result JSON
RESULT_JSON = {
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)
``` 

This script follows the workflow required for running the power flow and collecting the lowest voltage buses as you requested.