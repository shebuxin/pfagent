# required_dependencies: andes
import andes
import numpy as np

# Load the case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 15
idx_new_load = ssa.PQ.add(bus=15, name='PQ_VERIFY_035_B', p0=0.016, q0=0.011)

# Set the slack-bus voltage target to 1.030
slack_bus_index = ssa.Bus.idx.v[ssa.Bus.slack.v][0]
ssa.Vref.v[slack_bus_index] = 1.030

# Rerun power flow
andes.run_power_flow(ssa)

# Find maximum and minimum voltage buses
max_bus_index = np.argmax(ssa.Bus.v.v)
min_bus_index = np.argmin(ssa.Bus.v.v)

# Prepare the JSON output
RESULT_JSON = {
    "added_load_idx": idx_new_load,
    "max_bus": ssa.Bus.idx.v[max_bus_index],
    "max_voltage": round(ssa.Bus.v.v[max_bus_index], 6),
    "min_bus": ssa.Bus.idx.v[min_bus_index],
    "min_voltage": round(ssa.Bus.v.v[min_bus_index], 6),
    "total_pq_count": len(ssa.PQ.idx.v)
}

print("RESULT_JSON=", RESULT_JSON)