# required_dependencies: pandas, numpy, andes
import andes
import pandas as pd
import numpy as np

# Load the built-in IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set slack-bus voltage target
slack_bus = ssa.Bus.idx.v[ssa.Bus.slack.v][0]  # Get the index of the slack bus
ssa.V.set(target=1.020, idx=[slack_bus])

# Rerun power flow
andes.run_power_flow(ssa)

# Get voltage information
voltages = ssa.Bus.v[:]

# Determine slack bus voltage and count buses below 1.000 p.u.
slack_voltage = round(voltages[slack_bus - 1], 6)  # Subtract 1 for 0-based index
selected_count = np.sum(voltages < 1.000)

# Construct result JSON object
RESULT_JSON = {
    "slack_bus": int(slack_bus),
    "slack_setpoint": round(1.020, 6),
    "slack_voltage": slack_voltage,
    "selected_count": int(selected_count)
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")