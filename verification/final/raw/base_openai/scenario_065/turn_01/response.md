```python
# required_dependencies: andes
import andes

# Load the built-in Kundur case
case = andes.get_case("kundur/kundur_aw.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_powerflow(ssa)

# Report maximum and minimum voltages
max_voltage_idx = ssa.Bus.v.argmax()
min_voltage_idx = ssa.Bus.v.argmin()

# Prepare the result
result_json = {
    "max_bus": int(ssa.Bus.idx.v[max_voltage_idx]),
    "max_voltage": round(float(ssa.Bus.v[max_voltage_idx]), 6),
    "min_bus": int(ssa.Bus.idx.v[min_voltage_idx]),
    "min_voltage": round(float(ssa.Bus.v[min_voltage_idx]), 6),
}

# Print result
print(f"RESULT_JSON={result_json}")
```