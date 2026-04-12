# required_dependencies: andes
import andes
import os

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_pjm5_098.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow(ssa)

# Get slack bus info
slack_bus = ssa.Bus.idx[ssa.Bus.slack.v[0]]
slack_voltage = round(ssa.Bus.V.mag.v[slack_bus], 6)

# Get max and min voltage buses
max_bus = ssa.Bus.idx[np.argmax(ssa.Bus.V.mag.v)]
max_voltage = round(ssa.Bus.V.mag.v[max_bus], 6)

min_bus = ssa.Bus.idx[np.argmin(ssa.Bus.V.mag.v)]
min_voltage = round(ssa.Bus.V.mag.v[min_bus], 6)

# Create the result JSON
RESULT_JSON = {
    "slack_bus": int(slack_bus),
    "slack_voltage": float(slack_voltage),
    "max_bus": int(max_bus),
    "max_voltage": float(max_voltage),
    "min_bus": int(min_bus),
    "min_voltage": float(min_voltage)
}

# Print the result JSON
print(f'RESULT_JSON={RESULT_JSON}')