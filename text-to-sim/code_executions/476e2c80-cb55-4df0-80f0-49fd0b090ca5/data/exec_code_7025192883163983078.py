import andes
import numpy as np


ssa = andes.load(
    andes.get_case('ieee39/ieee39.xlsx'),
    setup=True,     # If setup=False, need to call ss.setup() before running the simulation
    no_output=True, 
    default_config=False
)

ssa.PFlow.run()


# Grab all voltage magnitudes (p.u.)
v_mag = np.abs(ssa.Bus.v.v)

# Find the index of the maximum voltage
idx_max = int(np.argmax(v_mag))

# Retrieve the bus idx from Bus sheet
bus_idx = ssa.Bus.idx.v
idx_max_sheet = bus_idx[idx_max]

# Retrieve the bus ID from BUS sheet
bus_ids = ssa.Bus.uid
max_bus = bus_ids[idx_max_sheet]

max_v = v_mag[idx_max]

print(f"Bus UID {max_bus} has the Highest Voltage: {max_v} p.u.")


# Load the IEEE 39-bus case, run a power-flow simulation, and identify the bus with the highest voltage magnitude, displaying its UID and value.