# required_dependencies: andes, matplotlib
import andes
import matplotlib.pyplot as plt

# Load the built-in IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.020 p.u.
pv_indices = ssa.PV.idx.v
if len(pv_indices) > 0:
    ssa.PV.Vset.v[0] = 1.020

# Scale every PQ load by 1.050
load_scale_factor = 1.050
ssa.PQ.P.v *= load_scale_factor
ssa.PQ.Q.v *= load_scale_factor

# Rerun the power flow analysis
andes.run_power_flow(ssa)

# Get bus voltages
voltages = ssa.Vmag.v
buses = ssa.Bus.idx.v

# Find min and max voltage buses
min_idx = voltages.argmin()
max_idx = voltages.argmax()

min_bus = buses[min_idx]
min_voltage = round(voltages[min_idx], 6)
max_bus = buses[max_idx]
max_voltage = round(voltages[max_idx], 6)

# Plot bar chart of bus voltages
plt.figure(figsize=(10, 6))
plt.bar(buses, voltages, color='blue')
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')
plt.title('Bus Voltages after Scaling PQ Loads')
plt.grid()

# Save the plot
plot_file = 'scenario_011_turn3_bar.png'
plt.savefig(plot_file)
plt.close()

# Prepare the RESULT_JSON
RESULT_JSON = {
    "scale_factor": load_scale_factor,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "plot_file": plot_file
}

# Print the result in the required format
print(f"RESULT_JSON={RESULT_JSON}")