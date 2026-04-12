# required_dependencies: andes
import andes
import os

# Load the case
case = os.path.join(os.getcwd(), "verify_pjm5_097.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_power_flow()

# Get voltage info
slack_bus = ssa.bus.idx.where(ssa.bus.slack.v)
slack_voltage = ssa.bus.v[slack_bus].round(6)

max_bus = ssa.bus.idx.where(ssa.bus.v == ssa.bus.v.max())
max_voltage = ssa.bus.v[max_bus].round(6)

min_bus = ssa.bus.idx.where(ssa.bus.v == ssa.bus.v.min())
min_voltage = ssa.bus.v[min_bus].round(6)

# Construct the RESULT_JSON
RESULT_JSON = {
    "slack_bus": int(slack_bus),
    "slack_voltage": float(slack_voltage),
    "max_bus": int(max_bus),
    "max_voltage": float(max_voltage),
    "min_bus": int(min_bus),
    "min_voltage": float(min_voltage)
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")