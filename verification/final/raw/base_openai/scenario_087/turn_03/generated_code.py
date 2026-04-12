# required_dependencies: andes
import andes
import numpy as np
import matplotlib.pyplot as plt

# Load the PJM 5-bus case
case = andes.get_case("5bus/pjm5bus.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Scale every PQ load by 1.050
scale_factor = 1.050
ssa.PQ.set('P', ssa.PQ.power.v * scale_factor)
ssa.PQ.set('Q', ssa.PQ.react.v * scale_factor)

# Add new PQ load at bus 3
new_load_idx = 'PQ_VERIFY_087_E'
ssa.PQ.add(idx=new_load_idx, bus=3, P0=0.016, Q0=0.01)

# Run the power flow
andes.run_power_flow(ssa)

# Extract bus voltages
bus_voltages = ssa.Bus.V.v  # Bus voltages
bus_ids = ssa.Bus.idx.v      # Bus IDs

# Find max and min voltage buses
max_voltage = np.max(bus_voltages)
min_voltage = np.min(bus_voltages)
max_bus = bus_ids[np.argmax(bus_voltages)]
min_bus = bus_ids[np.argmin(bus_voltages)]

# Save line plot of bus voltages
plt.figure()
plt.plot(bus_ids, bus_voltages, marker='o')
plt.title('Bus Voltages')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.savefig('scenario_087_turn3_voltage.png')
plt.close()

# Prepare result in JSON format
RESULT_JSON = {
    "added_load_idx": new_load_idx,
    "max_bus": int(max_bus),
    "max_voltage": round(max_voltage, 6),
    "min_bus": int(min_bus),
    "min_voltage": round(min_voltage, 6),
    "plot_file": 'scenario_087_turn3_voltage.png'
}

# Output the result
print(f"RESULT_JSON={RESULT_JSON}")