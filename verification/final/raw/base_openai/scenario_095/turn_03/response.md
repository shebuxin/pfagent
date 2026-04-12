```python
# required_dependencies: andes, matplotlib
import andes
import os
import matplotlib.pyplot as plt

# Load the uploaded case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_pjm5_095.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Scale every PQ load by 1.050
scale_factor = 1.050
ssa.PQ.load.v *= scale_factor

# Add a new PQ load at bus 3
added_load_idx = 'PQ_VERIFY_095_E'
ssa.PQ.add(bus=3, p0=0.016, q0=0.01, idx=added_load_idx)

# Rerun power flow
andes.run_power_flow()

# Extract bus voltages
bus_voltages = ssa.Bus.v_mag.v
bus_ids = ssa.Bus.idx.v

# Determine max and min bus voltages
max_voltage = max(bus_voltages)
min_voltage = min(bus_voltages)
max_bus = bus_ids[bus_voltages.index(max_voltage)]
min_bus = bus_ids[bus_voltages.index(min_voltage)]

# Save a line plot of bus voltages
plt.figure()
plt.plot(bus_ids, bus_voltages, marker='o')
plt.title("Bus Voltages")
plt.xlabel("Bus ID")
plt.ylabel("Voltage (pu)")
plt.grid()
plt.savefig('scenario_095_turn3_voltage.png')
plt.close()

# Print the result in the required format
RESULT_JSON = {
    "added_load_idx": added_load_idx,
    "max_bus": int(max_bus),
    "max_voltage": round(max_voltage, 6),
    "min_bus": int(min_bus),
    "min_voltage": round(min_voltage, 6),
    "plot_file": "scenario_095_turn3_voltage.png"
}
print("RESULT_JSON=", RESULT_JSON)
```