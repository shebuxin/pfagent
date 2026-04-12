```python
# required_dependencies: andes
import andes

case = andes.load(andes.get_case("kundur/kundur_full.xlsx"), setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.000
first_pv_index = 0  # Assuming the first PV is at index 0
case.PV.set(first_pv_index, V_target=1.000)

# Rerun power flow
result = andes.run_power_flow(case)

# Collecting results
pv_bus = case.PV.bus.v[first_pv_index]
pv_setpoint = 1.000
pv_voltage = round(result.V.mag.v[case.PV.bus.v[first_pv_index] - 1], 6)  # Adjusting for 0-based index
selected_count = sum(result.V.mag.v > 1.000)

RESULT_JSON = {
    "pv_bus": int(pv_bus),
    "pv_setpoint": float(pv_setpoint),
    "pv_voltage": float(pv_voltage),
    "selected_count": int(selected_count)
}

print("RESULT_JSON=", RESULT_JSON)
```