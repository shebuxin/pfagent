# required_dependencies: andes
import andes

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load
ssa.PQ.add(
    idx='PQ_VERIFY_004_A',
    bus=10,
    p0=0.019,
    q0=0.012
)

# Run power flow analysis again
andes.fpn(ssa)

# Define the voltage threshold
threshold = 1.015

# Get all bus voltages and their IDs
bus_voltages = ssa.Bus.V.v
bus_ids = ssa.Bus.idx.v

# Identify buses below the threshold
selected_bus_ids = [int(bus_ids[i]) for i in range(len(bus_voltages)) if bus_voltages[i] < threshold]
selected_bus_ids.sort()

# Find minimum voltage and corresponding bus
min_voltage = min(bus_voltages)
min_bus = int(bus_ids[bus_voltages.argmin()])

# Prepare the RESULT_JSON
RESULT_JSON = {
    'added_load_idx': 'PQ_VERIFY_004_A',
    'added_load_bus': 10,
    'threshold': round(threshold, 6),
    'selected_bus_ids': selected_bus_ids,
    'selected_count': len(selected_bus_ids),
    'min_bus': min_bus,
    'min_voltage': round(min_voltage, 6)
}

# Print the result in the specified format
print(f"RESULT_JSON={RESULT_JSON}")