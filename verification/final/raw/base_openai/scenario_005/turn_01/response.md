```python
# required_dependencies: pandas, numpy, andes
import andes
import pandas as pd
import numpy as np

# Load the built-in IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Get voltage information
voltages = ssa.Bus.v[:]
buses = np.arange(len(voltages)) + 1  # Bus IDs are 1-based

# Determine thresholds and filters
threshold = 1.015
selected_bus_ids = buses[voltages > threshold].tolist()
lowest_indices = np.argsort(voltages)[:2]
lowest_bus_ids = buses[lowest_indices].tolist()
lowest_voltages = np.round(voltages[lowest_indices].tolist(), 6)

# Construct result JSON object
RESULT_JSON = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": len(selected_bus_ids),
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")
```