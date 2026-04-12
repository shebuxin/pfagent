# required_dependencies: pandas, matplotlib
import andes
import pandas as pd
import matplotlib.pyplot as plt

# Load the built-in PJM 5-bus case
case = andes.load(andes.get_case("5bus/pjm5bus.json"), setup=True, no_output=True)

# Scale every PQ load by 1.040
scale_factor = 1.040
case.PQ.load.v *= scale_factor

# Add a new PQ load at bus 2
new_load_idx = 'PQ_VERIFY_086_E'
case.PQ.add(idx=new_load_idx, bus=2, p0=0.014, q0=0.008)

# Rerun power flow
case.run_power_flow()

# Get bus voltages
bus_voltages = case.Bus.v
bus_indices = case.Bus.idx.v

# Identify max and min voltages
max_voltage = bus_voltages.max()
min_voltage = bus_voltages.min()
max_bus = bus_indices[bus_voltages.argmax()]
min_bus = bus_indices[bus_voltages.argmin()]

# Save a line plot of bus voltages
plt.figure()
plt.plot(bus_indices, bus_voltages, marker='o', linestyle='-')
plt.title('Bus Voltages')
plt.xlabel('Bus Index')
plt.ylabel('Voltage (PU)')
plt.grid()
plt.savefig('scenario_086_turn3_voltage.png')

# Prepare the RESULT_JSON output
RESULT_JSON = {
    "added_load_idx": new_load_idx,
    "max_bus": int(max_bus),
    "max_voltage": round(max_voltage, 6),
    "min_bus": int(min_bus),
    "min_voltage": round(min_voltage, 6),
    "plot_file": 'scenario_086_turn3_voltage.png'
}

# Print the RESULT_JSON
print(f"RESULT_JSON={RESULT_JSON}")