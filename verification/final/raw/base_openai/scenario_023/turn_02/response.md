```python
# required_dependencies: numpy,json
import os
import numpy as np
import json
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_023.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target
first_pv_idx = ssa.PV.idx.v[0]  # Get index of the first PV
ssa.PV.v_target.set_value(first_pv_idx, 1.010)

# Rerun power flow
andes.power_flow(ssa)

# Get the affected PV bus voltage
pv_bus = int(ssa.PV.bus.v[first_pv_idx])
pv_voltage = float(ssa.PV.v.v[first_pv_idx])
pv_setpoint = 1.010  # Setpoint we just assigned

# Count how many buses are above 1.015 p.u.
bus_voltages = ssa.Bus.v.v
selected_count = int(np.sum(bus_voltages > 1.015))

# Prepare the RESULT_JSON output
result_json = {
    "pv_bus": pv_bus,
    "pv_setpoint": round(pv_setpoint, 6),
    "pv_voltage": round(pv_voltage, 6),
    "selected_count": selected_count
}

# Print the result
print(f'RESULT_JSON={json.dumps(result_json)}')
```