import andes

ssa = andes.load(
    andes.get_case('ieee39/ieee39.xlsx'),
    setup=False,     # If setup=False, need to call ss.setup() before running the simulation
    no_output=True, 
    default_config=False
)

# Add new load
ssa.add(model="PQ", param_dict=dict(bus=2, 
                                    idx="PQ_20", 
                                    p0=0.5, 
                                    q0=0.2
                                    ))

# Add new generator
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


# Load the IEEE 39-bus case, add a new load and a new PV generator, then set up and run a power-flow simulation.