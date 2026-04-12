# required_dependencies: andes,numpy
import andes
import numpy as np

# Load the uploaded case
case = "verify_ieee14_027.xlsx"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Inspect bus voltages
bus_voltages = ssa.Bus.v.v
bus_ids = ssa.Bus.idx.v

# Find the 3 buses with the lowest voltages
lowest_voltage_indices = np.argsort(bus_voltages)[:3]
selected_bus_ids = bus_ids[lowest_voltage_indices].tolist()
selected_voltages = np.round(bus_voltages[lowest_voltage_indices].tolist(), 6)

# Create RESULT_JSON
RESULT_JSON = {
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

# Print the results
print(f'RESULT_JSON={RESULT_JSON}')