```python
# required_dependencies: pandas, numpy
import os
import andes
import numpy as np

# Load the case from the uploaded Excel file
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_081.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.010
if ssa.PV.idx.v.size > 0:
    first_pv_index = ssa.PV.idx.v[0]
    ssa.PV.set(first_pv_index, "Vtarget", 1.010)

# Rerun power flow
andes.run_power_flow(ssa)

# Retrieve the affected PV bus voltage
affected_pv_voltage = ssa.PV.v[0]  # First PV's voltage

# Count how many buses are above 1.010 p.u.
buses_above_target = np.sum(ssa.bus.v > 1.010)

# Prepare the RESULT_JSON
result_json = {
    "pv_bus": int(ssa.PV.bus.v[0]),
    "pv_setpoint": round(1.010, 6),
    "pv_voltage": round(float(affected_pv_voltage), 6),
    "selected_count": int(buses_above_target)
}

# Print the result in the specified format
print(f"RESULT_JSON={result_json}")
```