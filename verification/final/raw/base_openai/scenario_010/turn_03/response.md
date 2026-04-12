```python
# required_dependencies: andes,json,matplotlib
import andes
import json
import matplotlib.pyplot as plt

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.015
pv_bus_idx = ssa.PV.idx.v[0]
ssa.PV.vset.v[pv_bus_idx] = 1.015

# Scale every PQ load by 1.040
scale_factor = 1.040
ssa.PQ.load.v *= scale_factor

# Rerun power flow
result = andes.run_power_flow(ssa)

# Get max and min voltage buses
max_voltage_idx = result.V.idx.v[result.V.mag.v.argmax()]
min_voltage_idx = result.V.idx.v[result.V.mag.v.argmin()]
max_voltage = round(result.V.mag.v[max_voltage_idx], 6)
min_voltage = round(result.V.mag.v[min_voltage_idx], 6)

# Save a bar chart of the bus voltages
plt.bar(result.V.idx.v, result.V.mag.v)
plt.title('Bus Voltages after Power Flow')
plt.xlabel('Bus Index')
plt.ylabel('Voltage (p.u.)')
plt.xticks(result.V.idx.v)
plt.grid()
plot_file = 'scenario_010_turn3_bar.png'
plt.savefig(plot_file)
plt.close()

# Prepare result JSON
result_json = {
    "scale_factor": float(scale_factor),
    "min_bus": int(min_voltage_idx),
    "min_voltage": float(min_voltage),
    "max_bus": int(max_voltage_idx),
    "max_voltage": float(max_voltage),
    "plot_file": plot_file
}

# Print result
print("RESULT_JSON=" + json.dumps(result_json))
```