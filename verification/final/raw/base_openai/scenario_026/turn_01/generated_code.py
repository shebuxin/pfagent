# required_dependencies: andes
import andes
import os
import numpy as np

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_026.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow analysis
andes.run_power_flow(ssa)

# Extract voltage data from buses
bus_ids = ssa.Bus.idx.v
voltages = ssa.Bus.V.v

# Combine bus IDs and voltages, then find the two buses with the lowest voltage
voltage_data = list(zip(bus_ids, voltages))
lowest_voltage_buses = sorted(voltage_data, key=lambda x: x[1])[:2]

# Prepare the result
selected_bus_ids = [int(bus[0]) for bus in lowest_voltage_buses]
selected_voltages = [round(float(bus[1]), 6) for bus in lowest_voltage_buses]

# Create RESULT_JSON
RESULT_JSON = {
    'selected_bus_ids': selected_bus_ids,
    'selected_voltages': selected_voltages
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)