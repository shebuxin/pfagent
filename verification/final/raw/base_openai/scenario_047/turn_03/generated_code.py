# required_dependencies: andes
import andes
import os

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_047.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target
slack_bus = ssa.BUS.idx.v[ssa.BUS.slack.v][0]
slack_setpoint = 1.015
ssa.Y[slack_bus] = slack_setpoint  # Update slack bus voltage target

# Add a new PQ load at bus 4
new_load_idx = 'PQ_VERIFY_047_B'
p0 = 0.012
q0 = 0.007
andes.PQ.add(ssa, idx=new_load_idx, bus=4, p0=p0, q0=q0)

# Run power flow
andes.run_power_flow(ssa)

# Report max-voltage bus, min-voltage bus, and total PQ loads
max_voltage_index = ssa.BUS.vm.v.argmax()
min_voltage_index = ssa.BUS.vm.v.argmin()

max_bus = ssa.BUS.idx.v[max_voltage_index]
max_voltage = round(ssa.BUS.vm.v[max_voltage_index], 6)
min_bus = ssa.BUS.idx.v[min_voltage_index]
min_voltage = round(ssa.BUS.vm.v[min_voltage_index], 6)
total_pq_count = len(ssa.PQ.idx.v)

# Prepare the result JSON object
RESULT_JSON = {
    "added_load_idx": new_load_idx,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "total_pq_count": total_pq_count,
}

# Print the result
print(f'RESULT_JSON={RESULT_JSON}')