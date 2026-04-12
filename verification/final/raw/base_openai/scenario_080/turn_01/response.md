```python
# required_dependencies: andes
import andes

# Load the uploaded case
case = "verify_kundur_080.xlsx"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
result = andes.run(ssa)

# Identify max and min voltage buses
max_voltage_bus_idx = ssa.bus.v[np.argmax(ssa.bus.v)]
min_voltage_bus_idx = ssa.bus.v[np.argmin(ssa.bus.v)]
max_voltage = round(ssa.bus.v.max(), 6)
min_voltage = round(ssa.bus.v.min(), 6)

# Prepare JSON result
RESULT_JSON = {
    "max_bus": int(max_voltage_bus_idx),
    "max_voltage": float(max_voltage),
    "min_bus": int(min_voltage_bus_idx),
    "min_voltage": float(min_voltage),
}

print("RESULT_JSON=", RESULT_JSON)
```