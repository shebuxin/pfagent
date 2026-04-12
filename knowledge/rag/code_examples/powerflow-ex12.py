import andes
import numpy as np


ssa = andes.load(
    andes.get_case('ieee39/ieee39.xlsx'),
    setup=True,     # If setup=False, need to call ss.setup() before running the simulation
    no_output=True,
)

ssa.PFlow.run()


# Grab all voltage magnitudes (p.u.)
v_mag = np.abs(ssa.Bus.v.v)

# Find the index of the maximum voltage
idx_max = int(np.argmax(v_mag))

bus_ids = ssa.Bus.idx.v
max_bus = bus_ids[idx_max]

max_v = v_mag[idx_max]

print(f"Bus {max_bus} has the Highest Voltage: {max_v:.4f} p.u.")


# Load the IEEE 39-bus case, run a power-flow simulation, and identify the bus with the highest voltage magnitude, displaying its UID and value.
