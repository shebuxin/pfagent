import andes

ssa = andes.load(
    andes.get_case('ieee39/ieee39.xlsx'),
    setup=False, no_output=True, preserve_case=True
)

# Add new load
ssa.add(model="PQ", param=dict(bus=1, idx="1", 
                                p0=0.5, q0=0.2))

# Add new generator
ssa.add(model="PV", param=dict(bus=14, idx='14', 
                                p0=0.5, q0=0.2, 
                                pmax=5, pmin=0, 
                                qmax=2, qmin=-2, 
                                v0=1.0, 
                                slack=False))

ssa.setup()
ssa.PFlow.run()