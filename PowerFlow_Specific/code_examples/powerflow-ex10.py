import andes
import numpy as np


ssa = andes.load(
    andes.get_case('ieee39/ieee39.xlsx'),
    setup=False,     # If setup=False, need to call ss.setup() before running the simulation
    no_output=True, 
    default_config=False
)

ssa.setup()
ssa.PFlow.run()


# Total number of generators
generators = np.concatenate((ssa.PV.pmin.v, ssa.Slack.pmin.v))

print("Toal Generators:", len(generators))


# Total number of load rows
loads = ssa.PQ.p0.v

print("Total Loads:", len(loads))


# Load the IEEE 39-bus case, run a power-flow simulation, and display the total number of generators and loads in the system.