```python
# required_dependencies: pandas, numpy
import andes
import numpy as np
import pandas as pd

# Load the IEEE 14 full case
ssa = andes.load(andes.get_case("ieee14/ieee14_full.xlsx"), setup=True, no_output=True, log=False)

# Run power flow analysis
andes.run(ssa)

# Count all buses above 1.025 p.u.
threshold = 1.025
voltage_levels = ssa.Bus.v
selected_bus_ids = np.where(voltage_levels > threshold)[0].tolist()
selected_count = len(selected_bus_ids)

# Get the two lowest-voltage buses
lowest_voltages_indices = np.argsort(voltage_levels)[:2]
lowest_bus_ids = lowest_voltages_indices.tolist()
lowest_voltages = [round(voltage_levels[i], 6) for i in lowest_voltages_indices]

# Prepare the JSON result
RESULT_JSON = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages
}

print(f"RESULT_JSON={RESULT_JSON}")
```