```python
# required_dependencies: andes
import andes

# Load the built-in IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow analysis
andes.run_power_flow(ssa)

# Inspect results for voltages
voltages = ssa.Vmag.v
buses = ssa.Bus.idx.v

# Find max and min voltage buses
max_idx = voltages.argmax()
min_idx = voltages.argmin()

max_bus = buses[max_idx]
max_voltage = round(voltages[max_idx], 6)
min_bus = buses[min_idx]
min_voltage = round(voltages[min_idx], 6)

# Prepare the RESULT_JSON
RESULT_JSON = {
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

# Print the result in the required format
print(f"RESULT_JSON={RESULT_JSON}")
```