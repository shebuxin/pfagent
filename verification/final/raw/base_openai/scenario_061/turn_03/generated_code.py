# required_dependencies: pandas
import andes
import os
import json

# Load the Kundur full case
script_dir = os.getcwd()
case = os.path.join(script_dir, "kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set slack bus voltage target
slack_bus_idx = ssa.Bus.idx.v[np.where(ssa.Bus.type.v == 'slack')[0][0]]  # Get index of slack bus
slack_setpoint = 0.990
ssa.V.set(slack_bus_idx, slack_setpoint)

# Add new PQ load at bus 4
new_load_idx = ssa.PQ.add(bus=4, name='PQ_VERIFY_061_B', p0=0.012, q0=0.007)

# Rerun power flow
andes.run_power_flow(ssa)

# Analyze max and min voltage buses
voltages = ssa.Bus.v.val
max_voltage = voltages.max()
min_voltage = voltages.min()
max_bus = ssa.Bus.idx.v[np.argmax(voltages)]
min_bus = ssa.Bus.idx.v[np.argmin(voltages)]

# Count total number of PQ loads
total_pq_count = len(ssa.PQ.idx.v)

# Create JSON result
result_json = {
    "added_load_idx": int(new_load_idx),
    "max_bus": int(max_bus),
    "max_voltage": round(max_voltage, 6),
    "min_bus": int(min_bus),
    "min_voltage": round(min_voltage, 6),
    "total_pq_count": total_pq_count
}

# Print the result
print("RESULT_JSON=" + json.dumps(result_json))