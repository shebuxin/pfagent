```python
# required_dependencies: numpy,pandas
import os
import andes
import numpy as np
import pandas as pd

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_044.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 10
bus_10_idx = np.where(ssa.Bus.id.v == 10)[0]
if len(bus_10_idx) == 0:
    raise ValueError("Bus 10 does not exist in the case.")
bus_10_idx = bus_10_idx[0]
ssa.PQ.add(idx='PQ_VERIFY_044_A', bus=ssa.Bus.id.v[bus_10_idx], p0=0.013, q0=0.008)

# Rerun power flow
andes.run_powerflow(ssa)

# Define threshold and find buses below it
threshold = 0.960
voltages = ssa.Bus.Vm.v
selected_bus_ids = ssa.Bus.id.v[voltages < threshold].tolist()
selected_count = len(selected_bus_ids)

# Find minimum voltage bus
min_voltage_idx = np.argmin(voltages)
min_bus = ssa.Bus.id.v[min_voltage_idx]
min_voltage = round(voltages[min_voltage_idx], 6)

# Prepare JSON result
RESULT_JSON = {
    "added_load_idx": 'PQ_VERIFY_044_A',
    "added_load_bus": int(ssa.PQ.bus.v[np.where(ssa.PQ.idx.v == 'PQ_VERIFY_044_A')[0][0]]),
    "threshold": float(threshold),
    "selected_bus_ids": sorted(selected_bus_ids),
    "selected_count": int(selected_count),
    "min_bus": int(min_bus),
    "min_voltage": float(min_voltage),
}

# Print result
print(f"RESULT_JSON={RESULT_JSON}")
```