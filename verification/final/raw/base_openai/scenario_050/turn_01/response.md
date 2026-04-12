```python
# required_dependencies: numpy, pandas, json
import os
import numpy as np
import pandas as pd
import json
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_050.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run(ssa)

# Extracting bus voltages
bus_voltages = ssa.Bus.v.v  # Assuming this contains the bus voltage values
threshold = 1.050

# Finding buses above threshold
selected_bus_ids = np.where(bus_voltages > threshold)[0].tolist()
selected_count = len(selected_bus_ids)

# Finding the two lowest voltage buses
lowest_bus_ids = np.argsort(bus_voltages)[:2].tolist()
lowest_voltages = [round(bus_voltages[i], 6) for i in lowest_bus_ids]

# Constructing the result JSON
result_json = {
    "threshold": round(threshold, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_count": selected_count,
    "lowest_bus_ids": lowest_bus_ids,
    "lowest_voltages": lowest_voltages,
}

# Printing the result
print("RESULT_JSON=" + json.dumps(result_json))
```