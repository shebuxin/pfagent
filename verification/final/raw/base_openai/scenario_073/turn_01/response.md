Here's a complete runnable Python script following your requirements. Make sure you have the ANDES library installed and properly set up in your environment to execute this script.

```python
# required_dependencies: andes
import andes
import os

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_073.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run(ssa)

# Extracting slack bus information
slack_bus_idx = ssa.BUS.idx.v[ssa.BUS.slack.v]
slack_voltage = ssa.BUS.vmag.v[slack_bus_idx]

# Getting all bus voltages and their IDs
bus_ids = ssa.BUS.id.v
bus_voltages = ssa.BUS.vmag.v

# Pairing bus IDs with voltages and sorting to get top 4 highest voltages
sorted_buses = sorted(zip(bus_ids, bus_voltages), key=lambda x: x[1], reverse=True)
top_buses = sorted_buses[:4]

# Preparing the output format
selected_bus_ids = [bus[0] for bus in top_buses]
selected_voltages = [round(bus[1], 6) for bus in top_buses]

# Result JSON
RESULT_JSON = {
    "slack_bus": int(slack_bus_idx),
    "slack_voltage": round(slack_voltage, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")
``` 

Make sure you have your environment set up correctly for ANDES and have the necessary Python dependencies installed. This script will load your specified case, run the power flow analysis, and provide the results as requested.