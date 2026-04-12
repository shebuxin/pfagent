import andes
import numpy as np


ssa = andes.load(
    andes.get_case('kundur/kundur_full.xlsx')
)

ssa.PFlow.run()


bus_angles = ssa.Bus.a.v
bus_ids = ssa.Bus.idx.v

# Max and min voltages
index_max = int(np.nanargmax(bus_angles))
index_min = int(np.nanargmin(bus_angles))
gap = bus_angles[index_max] - bus_angles[index_min]

print(f"Max Angle: Bus {bus_ids[index_max]} = {bus_angles[index_max]:.4f} Radians")
print(f"Min Angle: Bus {bus_ids[index_min]} = {bus_angles[index_min]:.4f} Radians")

print(f"Gap: {gap} Radians")

if gap > (np.pi / 4):
    print("Incorrect: Exceeds Upper Bound")
else:
    print("Correct: Within Upper Bound")


# Check for bus angle difference, make sure it's 1/4 of pi (maximum), normal value is 30 degrees