# required_dependencies: andes
import andes

# Load the built-in PJM 5-bus case
case = andes.load(andes.get_case("5bus/pjm5bus.json"), setup=True, no_output=True, log=False)

# Run power flow
case.run_power_flow()

# Extract voltages and bus indices
voltages = case.bus.v
slack_bus = case.bus.idx.slack
slack_voltage = round(voltages[slack_bus], 6)

# Find max and min voltage buses
max_voltage = round(max(voltages), 6)
min_voltage = round(min(voltages), 6)
max_bus = int(next(i for i, v in enumerate(voltages) if v == max_voltage))
min_bus = int(next(i for i, v in enumerate(voltages) if v == min_voltage))

# Prepare the result JSON
RESULT_JSON = {
    "slack_bus": int(slack_bus),
    "slack_voltage": slack_voltage,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
}

# Print the result JSON
print(f"RESULT_JSON={RESULT_JSON}")