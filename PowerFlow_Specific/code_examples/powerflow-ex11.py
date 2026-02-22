import andes
import numpy as np


ssa = andes.load(
    andes.get_case('ieee39/ieee39.xlsx'),
    setup=True,     # If setup=False, need to call ss.setup() before running the simulation
    no_output=True, 
    default_config=False
)

ssa.PFlow.run()


generator_capacity = np.concatenate([ssa.PV.Sn.v, ssa.Slack.Sn.v])  # capacity for each generator

# Compute overall extremes
max_capacity = np.max(generator_capacity)
min_capacity = np.min(generator_capacity)

print(f"Maximum Generator Capacity: {max_capacity} MVA")
print(f"Minimum Generator Capacity: {min_capacity} MVA")


# Load the IEEE 39-bus case, run a power-flow simulation, and display the maximum and minimum generator capacities in the system.