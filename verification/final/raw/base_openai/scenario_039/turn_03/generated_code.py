# required_dependencies: andes, matplotlib
import andes
import matplotlib.pyplot as plt

# Load the IEEE 39 case
case = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.015
first_pv_idx = case.PV.idx.v[0]
case.PV.voltage_target.set(first_pv_idx, 1.015)

# Scale every PQ load by 1.040
case.PQ.load.set(case.PQ.idx.v, case.PQ.load.v * 1.040)

# Rerun power flow analysis
case.solve_power_flow()

# Extract voltage information
voltages = case.Bus.v
buses = case.Bus.idx

min_voltage_idx = voltages.argmin()
max_voltage_idx = voltages.argmax()

min_bus = int(buses[min_voltage_idx])
min_voltage = round(float(voltages[min_voltage_idx]), 6)
max_bus = int(buses[max_voltage_idx])
max_voltage = round(float(voltages[max_voltage_idx]), 6)

# Save bar chart of bus voltages
plt.figure(figsize=(12, 6))
plt.bar(buses, voltages, color='blue')
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')
plt.title('Bus Voltages after Power Flow')
plt.grid()
plt.savefig('scenario_039_turn3_bar.png')
plt.close()

# Create the JSON result
RESULT_JSON = {
    "scale_factor": 1.040,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "plot_file": 'scenario_039_turn3_bar.png'
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")