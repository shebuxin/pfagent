# required_dependencies: andes

import andes

ssa = andes.load(
    andes.get_case('ieee39/ieee39.xlsx'),
    setup=True,     # no setup menu
    no_output=True,   # no print output
    log=False        # no logging
)

# Run the power flow
ssa.PFlow.run()

# Extract voltage magnitudes at all buses
v_mag = ssa.Bus.v.v

# Find the index of the bus with the maximum voltage
max_idx = v_mag.argmax()

# Get the bus ID with the maximum voltage
bus_id = ssa.Bus.idx.v[max_idx]

# Get the voltage magnitude for that bus
max_v = v_mag[max_idx]

print(f"Bus {bus_id} has the Highest Voltage: {max_v:.4f} p.u.")