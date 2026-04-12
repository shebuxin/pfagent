# required_dependencies: andes
import andes
import os
import json

# Load the case from the uploaded file
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_055.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow analysis
andes.run_power_flow()

# Get the bus voltages and sort them
bus_ids = ssa.Bus.idx.v
voltages = ssa.Bus.vmag.v
sorted_indices = sorted(range(len(voltages)), key=lambda i: voltages[i])
lowest_voltage_buses = sorted_indices[:3]

# Prepare the result
selected_bus_ids = [int(bus_ids[i]) for i in lowest_voltage_buses]
selected_voltages = [round(float(voltages[i]), 6) for i in lowest_voltage_buses]

# Print the result as JSON
RESULT_JSON = json.dumps({
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
})
print(f"RESULT_JSON={RESULT_JSON}")