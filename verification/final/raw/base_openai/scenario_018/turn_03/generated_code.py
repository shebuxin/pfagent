# required_dependencies: numpy,pandas,matplotlib
import os
import andes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the case from the uploaded file
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_018.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Keep the previously added PQ load at bus 10
load_idx = 'PQ_VERIFY_018_A'
bus_number = 10
p0 = 0.019
q0 = 0.012
bus_mask = np.where(ssa.Bus.idx.v == bus_number)[0]

if bus_mask.size == 0:
    raise ValueError(f"No bus found with number {bus_number}")

# Adding the PQ load if it doesn't exist
if not load_idx in ssa.PQ.idx.v:
    ssa.PQ.set(load_idx, bus=bus_mask[0], p0=p0, q0=q0)

# Set scale factor for PQ loads
scale_factor = 1.060
ssa.PQ.p0 *= scale_factor
ssa.PQ.q0 *= scale_factor

# Run power flow
andes.run_power_flow(ssa)

# Gather bus voltage data
bus_voltages = ssa.Bus.Vmag.v
bus_ids = ssa.Bus.idx.v

# Find max and min voltage buses
max_index = np.argmax(bus_voltages)
min_index = np.argmin(bus_voltages)

max_bus = bus_ids[max_index]
max_voltage = round(bus_voltages[max_index], 6)
min_bus = bus_ids[min_index]
min_voltage = round(bus_voltages[min_index], 6)

# Plot bus voltage magnitude
plt.figure()
plt.plot(bus_ids, bus_voltages, marker='o')
plt.title('Bus Voltage Magnitude')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plot_file = 'scenario_018_turn3_line.png'
plt.savefig(plot_file)
plt.close()

# Prepare result JSON
RESULT_JSON = {
    "scale_factor": float(scale_factor),
    "max_bus": int(max_bus),
    "max_voltage": float(max_voltage),
    "min_bus": int(min_bus),
    "min_voltage": float(min_voltage),
    "plot_file": plot_file
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)