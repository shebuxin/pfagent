# required_dependencies: andes, matplotlib
import os
import andes
import matplotlib.pyplot as plt

# Load the case
case = os.path.join(os.getcwd(), "verify_ieee14_024.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target
first_pv_idx = ssa.PV.idx.v[0]
ssa.PV.set(first_pv_idx, Vtarget=1.015)

# Scale every PQ load by 1.040
scale_factor = 1.040
ssa.PQ.set(ssa.PQ.idx.v, P=ssa.PQ.P.v * scale_factor, Q=ssa.PQ.Q.v * scale_factor)

# Run power flow again
andes.run(ssa, solver='default', log=True)

# Analyze bus voltage results
voltages = ssa.bus.v
min_voltage_idx = voltages.argmin()
max_voltage_idx = voltages.argmax()

min_bus = ssa.bus.idx.v[min_voltage_idx]
min_voltage = round(voltages[min_voltage_idx], 6)
max_bus = ssa.bus.idx.v[max_voltage_idx]
max_voltage = round(voltages[max_voltage_idx], 6)

# Save a bar chart of the bus voltages
plt.bar(ssa.bus.idx.v, voltages)
plt.xlabel("Bus Index")
plt.ylabel("Bus Voltage (p.u.)")
plt.title("Bus Voltages after Power Flow")
plt.savefig("scenario_024_turn3_bar.png")
plt.close()

# Prepare results
RESULT_JSON = {
    "scale_factor": float(scale_factor),
    "min_bus": int(min_bus),
    "min_voltage": float(min_voltage),
    "max_bus": int(max_bus),
    "max_voltage": float(max_voltage),
    "plot_file": "scenario_024_turn3_bar.png"
}

# Print result in specified JSON format
print(f"RESULT_JSON={RESULT_JSON}")