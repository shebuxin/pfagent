# required_dependencies: pandas, numpy
import andes
import pandas as pd
import numpy as np

# Load the Kundur full case
case = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the slack-bus voltage target to 1.000
slack_bus_index = ssa.Bus.idx.v[0]  # Assuming the slack bus is the first one
ssa.Bus.voltage_target.set(slack_bus_index, 1.000)

# Add a new PQ load at bus 6
load_idx = ssa.PQ.add("PQ_VERIFY_062_B", bus=6, p0=0.014, q0=0.009) 

# Rerun power flow
andes.run_power_flow(ssa)

# Get the maximum and minimum voltage buses
max_voltage = np.max(ssa.Bus.voltage.v)
min_voltage = np.min(ssa.Bus.voltage.v)
max_bus_idx = ssa.Bus.idx.v[np.argmax(ssa.Bus.voltage.v)]
min_bus_idx = ssa.Bus.idx.v[np.argmin(ssa.Bus.voltage.v)]

# Count total number of PQ loads present
total_pq_count = len(ssa.PQ.idx.v)

# Prepare RESULT_JSON
RESULT_JSON = {
    "added_load_idx": load_idx,
    "max_bus": max_bus_idx,
    "max_voltage": round(max_voltage, 6),
    "min_bus": min_bus_idx,
    "min_voltage": round(min_voltage, 6),
    "total_pq_count": total_pq_count
}

# Print result
print(f"RESULT_JSON={RESULT_JSON}")