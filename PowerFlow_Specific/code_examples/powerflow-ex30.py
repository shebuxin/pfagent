import andes

ssa = andes.load(
    andes.get_case('GBnetwork/GBnetwork.xlsx'),
    setup=False,     # If setup=False, need to call ss.setup() before running the simulation
    no_output=True, 
    default_config=False
)

# Add new load
ssa.add(model="PQ", param_dict=dict(bus=2, 
                                    idx="PQ_484", 
                                    p0=60/ssa.config.mva, 
                                    q0=100/ssa.config.mva
                                    ))

ssa.setup()
ssa.PFlow.run()

print("PQ Bus Active Power", ssa.PQ.p0.v)
print("PQ Bus Reactive Power", ssa.PQ.q0.v)
print("PQ Table", ssa.PQ.as_df())


# Load the IEEE 39-bus case, add a new load with 60 MW p0 and 100 MW q0, then set up and run a power-flow simulation.