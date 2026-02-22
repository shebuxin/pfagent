import andes
import numpy as np


ssa = andes.load(
    andes.get_case('ieee39/ieee39.xlsx'),
    setup=True,     # If setup=False, need to call ss.setup() before running the simulation
    no_output=True, 
    default_config=False
)

ssa.PFlow.run()


# Grab all bus voltage angles (radians)
angles_rad = ssa.Bus.a.v

# Find the buses with the maximum and minimum angles
idx_max = int(np.argmax(angles_rad))
idx_min = int(np.argmin(angles_rad))

# Convert positions -> sheet indices -> UIDs
bus_idx = ssa.Bus.idx.v  # mapping from position -> sheet index
bus_ids = ssa.Bus.uid  # array of bus UIDs keyed by sheet index
sheet_max = bus_idx[idx_max]
sheet_min = bus_idx[idx_min]
bus_max = bus_ids[sheet_max]
bus_min = bus_ids[sheet_min]

max_angle = angles_rad[idx_max]
min_angle = angles_rad[idx_min]
diff_angle = max_angle - min_angle

print(f"Bus UID {bus_max} has the Largest Angle: {max_angle} Radians")
print(f"Bus UID {bus_min} has the Smallest Angle: {min_angle} Radians")
print(f"Maximum Bus Angle Difference is Between {bus_max} and {bus_min}: {diff_angle} Radians")


# Load the IEEE 39-bus case, run a power-flow simulation, and identify the buses with the largest and smallest voltage angles (in radians), displaying their UIDs, values, and the maximum angle difference.