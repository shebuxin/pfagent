```python
# required_dependencies: numpy,pandas,andes
import numpy as np
import pandas as pd
import os
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_052.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Modify the first PV's voltage target
first_pv_idx = ssa.PV.idx.v[0]  # Get the index of the first PV
ssa.PV.v_target.set(first_pv_idx, 1.010)

# Run power flow analysis again
result = andes.run_power_flow(ssa)

# Get details for the affected PV bus voltage
pv_bus = ssa.PV.bus.v[first_pv_idx]
pv_setpoint = round(float(ssa.PV.v_target.v[first_pv_idx]), 6)
pv_voltage = round(float(ssa.PV.v.v[first_pv_idx]), 6)

# Count how many buses are above 1.030 p.u.
above_count = np.sum(ssa.Bus.v.v > 1.030)

# Prepare the result JSON
RESULT_JSON = {
    "pv_bus": int(pv_bus),
    "pv_setpoint": pv_setpoint,
    "pv_voltage": pv_voltage,
    "selected_count": int(above_count)
}

print(f"RESULT_JSON={RESULT_JSON}")
```