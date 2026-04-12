import andes
import numpy as np

ssa = andes.load(
    andes.get_case('ieee39/ieee39.xlsx'),
    setup=True,
    no_output=True,
)

ssa.PFlow.run()


# Grab all bus voltage angles (radians)
angles_rad = ssa.Bus.a.v
angles_deg = np.degrees(angles_rad)

# Find the buses with the maximum and minimum angles
idx_max = int(np.argmax(angles_deg))
idx_min = int(np.argmin(angles_deg))

bus_ids = ssa.Bus.idx.v
bus_max = bus_ids[idx_max]
bus_min = bus_ids[idx_min]

max_angle = angles_deg[idx_max]
min_angle = angles_deg[idx_min]
diff_angle = max_angle - min_angle

print(f"Bus {bus_max} has the Largest Angle: {max_angle:.4f} deg")
print(f"Bus {bus_min} has the Smallest Angle: {min_angle:.4f} deg")
print(f"Maximum Bus Angle Difference is Between {bus_max} and {bus_min}: {diff_angle:.4f} deg")


# Load the IEEE 39-bus case, run a power-flow simulation, and identify the buses with the largest and smallest voltage angles in degrees, displaying their UIDs, values, and the maximum angle difference.
