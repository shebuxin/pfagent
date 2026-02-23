# required_dependencies: andes, numpy

import andes
import numpy as np

ssa = andes.load(
    andes.get_case('ieee39/ieee39.xlsx'),
    setup=False, no_output=True, default_config=False
)

ssa.setup()
ssa.PFlow.run()

# Extract all bus voltage magnitudes (p.u.)
v_mag = np.abs(ssa.Bus.v.v)

# Find indices of min and max voltages
idx_min = int(np.argmin(v_mag))
idx_max = int(np.argmax(v_mag))

# Convert indices to bus IDs
bus_min = ssa.Bus.idx.v[idx_min]
bus_max = ssa.Bus.idx.v[idx_max]

# Get voltage values
v_min = v_mag[idx_min]
v_max = v_mag[idx_max]

# Compute difference
diff = v_max - v_min

print(f"Bus {bus_min} has the Lowest Voltage: {v_min:.4f} p.u.")
print(f"Bus {bus_max} has the Highest Voltage: {v_max:.4f} p.u.")
print(f"Maximum Bus Voltage Difference: {diff:.4f} p.u.")

print("\nBus Voltages:")
for bus_id, voltage in zip(ssa.Bus.idx.v, ssa.Bus.v):
    print(f"Bus {bus_id}: {voltage:.4f} p.u.")


# Compute and display the maximum bus voltage difference (corrected)