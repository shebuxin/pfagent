# required_dependencies: andes, matplotlib
import andes
import matplotlib.pyplot as plt

# Load the case
case = "verify_ieee39_051.xlsx"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.005
first_pv_idx = ssa.PV.idx.v[0]
ssa.PV.v_target.set(first_pv_idx, 1.005)

# Scale every PQ load by 1.020
scale_factor = 1.020
ssa.PQ.P.set(slice(None), ssa.PQ.P.v * scale_factor)
ssa.PQ.Q.set(slice(None), ssa.PQ.Q.v * scale_factor)

# Rerun power flow
andes.run_power_flow(ssa)

# Gather bus voltages for the result
voltages = ssa.Bus.v
min_bus_index = voltages.argmin()
max_bus_index = voltages.argmax()

min_bus = ssa.Bus.idx.v[min_bus_index]
min_voltage = round(voltages[min_bus_index], 6)
max_bus = ssa.Bus.idx.v[max_bus_index]
max_voltage = round(voltages[max_bus_index], 6)

# Plot the bus voltages as a bar chart
plt.figure(figsize=(10, 6))
plt.bar(range(len(voltages)), voltages)
plt.title('Bus Voltages')
plt.xlabel('Bus Index')
plt.ylabel('Voltage (p.u.)')
plt.xticks(range(len(voltages)), ssa.Bus.idx.v, rotation=45)
plt.grid()
plot_file = 'scenario_051_turn3_bar.png'
plt.tight_layout()
plt.savefig(plot_file)
plt.close()

# Prepare the result
RESULT_JSON = {
    "scale_factor": float(scale_factor),
    "min_bus": int(min_bus),
    "min_voltage": float(min_voltage),
    "max_bus": int(max_bus),
    "max_voltage": float(max_voltage),
    "plot_file": plot_file,
}

# Print the result in the required format
print(f'RESULT_JSON={RESULT_JSON}')