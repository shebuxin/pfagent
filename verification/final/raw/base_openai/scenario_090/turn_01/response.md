```python
# required_dependencies: andes
import andes

# Load the built-in PJM 5-bus case 
case = andes.get_case("5bus/pjm5bus.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_powerflow(ssa)

# Extract voltage results
voltage = ssa.Bus.V.v
slack_idx = ssa.Bus.idx.slack
slack_voltage = round(voltage[slack_idx], 6)

# Identify max and min voltage buses
max_voltage_idx = voltage.argmax()
min_voltage_idx = voltage.argmin()

max_bus = ssa.Bus.idx.v[max_voltage_idx]
min_bus = ssa.Bus.idx.v[min_voltage_idx]
max_voltage = round(voltage[max_voltage_idx], 6)
min_voltage = round(voltage[min_voltage_idx], 6)

# Prepare the result
result = {
    "slack_bus": int(max_bus),
    "slack_voltage": slack_voltage,
    "max_bus": int(max_bus),
    "max_voltage": max_voltage,
    "min_bus": int(min_bus),
    "min_voltage": min_voltage
}

# Print the result in the required format
print(f"RESULT_JSON={result}")
```