# required_dependencies: pandas
import andes
import numpy as np

# Load the Kundur full case
case = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True)

# Add a new PQ load at bus 7
bus_7_idx = np.where(ssa.Bus.idx.v == 7)[0][0]
ssa.PQ.add(idx='PQ_VERIFY_059_A', bus=bus_7_idx, p0=0.016, q0=0.01)

# Setup the modified case
andes.setup(ssa)

# Run power flow again
andes.solve()

# Extract updated bus data
voltages = ssa.Bus.v.v
bus_ids = ssa.Bus.idx.v

# Define the threshold
threshold = 0.960

# Find buses below threshold
selected_bus_ids = bus_ids[voltages < threshold].tolist()
min_bus_idx = np.argmin(voltages)
min_voltage = round(voltages[min_bus_idx], 6)
min_bus = int(ssa.Bus.idx.v[min_bus_idx])

# Prepare the result JSON
RESULT_JSON = {
    "added_load_idx": "PQ_VERIFY_059_A",
    "added_load_bus": bus_7_idx,
    "threshold": threshold,
    "selected_bus_ids": sorted(selected_bus_ids),
    "selected_count": len(selected_bus_ids),
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

print(f"RESULT_JSON={RESULT_JSON}")