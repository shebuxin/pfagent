```python
# required_dependencies: pandas
import os
import andes
import pandas as pd

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_074.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 9
new_load_idx = 'PQ_VERIFY_074_A'
bus_9_idx = np.where(ssa.Bus.idx.v == 9)[0][0]  # Find the index for bus 9
ssa.PQ.add(new_load_idx, bus_9_idx, p0=0.019, q0=0.012)

# Run power flow after adding the load
andes.run_powerflow(ssa)

# Extract voltage data
voltages = ssa.Bus.v.v

# Find buses below 0.970 p.u. and the minimum-voltage bus
threshold = 0.970
selected_bus_ids = ssa.Bus.idx.v[voltages < threshold].tolist()
min_bus_idx = np.argmin(voltages)
min_bus = int(ssa.Bus.idx.v[min_bus_idx])
min_voltage = round(float(voltages[min_bus_idx]), 6)

# Prepare the result JSON
RESULT_JSON = {
    "added_load_idx": new_load_idx,
    "added_load_bus": 9,
    "threshold": threshold,
    "selected_bus_ids": sorted(selected_bus_ids),
    "selected_count": len(selected_bus_ids),
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)
```