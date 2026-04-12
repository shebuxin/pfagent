```python
# required_dependencies: andes
import andes
import numpy as np

# Load IEEE 39 case
case = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True)

# Set the first PV voltage target to 1.010
case.PV.set_targets([1.010])

# Rerun power flow
case.run()

# Inspect results
pv_bus = int(case.PV.bus[0])
pv_setpoint = float(case.PV.target.v[0])
pv_voltage = float(case.V[np.where(case.bus == pv_bus)])

# Count how many buses are above 1.030 p.u.
selected_count = int(np.sum(case.V > 1.030))

# Prepare result JSON
result_json = {
    "pv_bus": pv_bus,
    "pv_setpoint": round(pv_setpoint, 6),
    "pv_voltage": round(pv_voltage, 6),
    "selected_count": selected_count
}

# Print the result
print("RESULT_JSON=", result_json)
```