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


print("Before:")

# Total number of generators
generators = np.concatenate([ssa.PV.pmin.v, ssa.Slack.pmin.v])

print("Toal Generators:", len(generators))


# Total number of load rows
loads = ssa.PQ.p0.v

print("Total Loads:", len(loads))


ssa = andes.load(
    andes.get_case('ieee39/ieee39.xlsx'),
    setup=False,     # If setup=False, need to call ss.setup() before running the simulation
    no_output=True, 
    default_config=False
)

# Add new load
# Remove load can be done by setting the p0 and q0 to 0
ssa.add(model="PQ", param_dict=dict(bus=2, 
                                    idx="PQ_20", 
                                    p0=0.5, 
                                    q0=0.2
                                    ))

# Add new generator
# Remove generator can be done by setting the p0 and q0 to 0
ssa.add(model="PV", param_dict=dict(bus=15, 
                                    idx='15', 
                                    p0=0.5, 
                                    q0=0.2,
                                    pmax=5,
                                    pmin=0,
                                    qmax=2,
                                    qmin=-2,
                                    v0=1.0,
                                    ))

ssa.setup()
ssa.PFlow.run()


print("After:")

# Total number of generators
generators = np.concatenate([ssa.PV.pmin.v, ssa.Slack.pmin.v])

print("Toal Generators:", len(generators))


# Total number of load rows
loads = ssa.PQ.p0.v

print("Total Loads:", len(loads))


# Load the IEEE 39-bus case, run a power-flow simulation, and display the total numbers of generators and loads before and after adding a new load and a new PV generator.