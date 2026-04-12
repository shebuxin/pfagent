# required_dependencies: andes
import andes

# Load the Kundur full case
case = andes.get_case("kundur/kundur_full.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 4
load_idx = ssa.PQ.add("PQ_VERIFY_057_A", bus=4, p0=0.01, q0=0.006)

# Run the power flow
andes.run_power_flow(ssa)

# Gather voltage data
bus_voltages = ssa.Bus.v.v
bus_ids = ssa.Bus.idx.v

# Define threshold for voltage level
threshold = 0.94

# Find buses below the threshold
selected_buses = [(bus_ids[i], bus_voltages[i]) for i in range(len(bus_voltages)) if bus_voltages[i] < threshold]
selected_bus_ids = sorted(bus[0] for bus in selected_buses)

# Find the minimum voltage bus
min_voltage_data = min(zip(bus_ids, bus_voltages), key=lambda x: x[1])
min_bus = min_voltage_data[0]
min_voltage = round(min_voltage_data[1], 6)

# Prepare the result JSON
RESULT_JSON = {
    "added_load_idx": load_idx,
    "added_load_bus": 4,
    "threshold": threshold,
    "selected_bus_ids": selected_bus_ids,
    "selected_count": len(selected_bus_ids),
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")