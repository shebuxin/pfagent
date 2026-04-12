# required_dependencies: andes,json
import andes
import json

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True)

# Set the slack-bus voltage target
slack_bus_idx = ssa.Bus.slack.v[0]
slack_setpoint = 1.040
ssa.Bus.v.target[slack_bus_idx] = slack_setpoint

# Add new PQ load at bus 10
bus_10_idx = 9  # bus indices are typically 0-based in data structures
new_load_idx = ssa.PQ.add(bus=10, p0=0.018, q0=0.013, tag='PQ_VERIFY_008_B')

# Rerun power flow
andes.run(ssa, no_output=True)

# Determine max and min voltage buses
bus_voltages = ssa.Bus.v.v
max_voltage = max(bus_voltages)
min_voltage = min(bus_voltages)
max_bus = ssa.Bus.idx.v[bus_voltages.argmax()]
min_bus = ssa.Bus.idx.v[bus_voltages.argmin()]

# Count total PQ loads
total_pq_count = len(ssa.PQ.idx.v)

# Prepare the result JSON
result_json = {
    "added_load_idx": int(new_load_idx),
    "max_bus": int(max_bus),
    "max_voltage": round(max_voltage, 6),
    "min_bus": int(min_bus),
    "min_voltage": round(min_voltage, 6),
    "total_pq_count": total_pq_count
}

# Print the result in the required format
print(f"RESULT_JSON={json.dumps(result_json)}")