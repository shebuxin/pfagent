# required_dependencies: andes
import andes
import os
import json

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_054.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Identify the buses and their voltages
bus_ids = ssa.Bus.idx.v
bus_voltages = ssa.Bus.vmag.v

# Combine bus IDs and voltages
bus_voltage_pairs = list(zip(bus_ids, bus_voltages))

# Sort by voltage and select the two lowest-voltage buses
sorted_buses = sorted(bus_voltage_pairs, key=lambda pair: pair[1])[:2]
selected_bus_ids = [bus[0] for bus in sorted_buses]
selected_voltages = [round(bus[1], 6) for bus in sorted_buses]

# Prepare the result json output
result_json = {
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

# Print the result
print(f"RESULT_JSON={json.dumps(result_json)}")