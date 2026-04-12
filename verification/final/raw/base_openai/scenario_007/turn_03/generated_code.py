# required_dependencies: pandas, numpy
import andes
import numpy as np
import pandas as pd

# Load the IEEE 14 full case
ssa = andes.load(andes.get_case("ieee14/ieee14_full.xlsx"), setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus = ssa.Slack.idx.v[0]
slack_setpoint = 1.035
ssa.Slack.v_target.set(slack_setpoint)

# Add a new PQ load at bus 9
added_load_idx = "PQ_VERIFY_007_B"
p0 = 0.016
q0 = 0.011
bus_number = 9
ssa.PQ.add(bus=bus_number, idx=added_load_idx, p0=p0, q0=q0)

# Run power flow analysis
andes.run(ssa)

# Get the max and min voltage buses and total PQ loads count
voltage_levels = ssa.Bus.v
max_bus = np.argmax(voltage_levels)
min_bus = np.argmin(voltage_levels)
max_voltage = round(voltage_levels[max_bus], 6)
min_voltage = round(voltage_levels[min_bus], 6)
total_pq_count = len(ssa.PQ.idx.v)

# Prepare the JSON result
RESULT_JSON = {
    "added_load_idx": added_load_idx,
    "max_bus": int(max_bus),
    "max_voltage": max_voltage,
    "min_bus": int(min_bus),
    "min_voltage": min_voltage,
    "total_pq_count": total_pq_count
}

print(f"RESULT_JSON={RESULT_JSON}")