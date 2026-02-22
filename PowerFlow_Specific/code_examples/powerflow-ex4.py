import andes
import numpy as np

ssa = andes.load(
    andes.get_case('ieee39/ieee39.xlsx'),
    setup=False,     # If setup=False, need to call ss.setup() before running the simulation
    no_output=True, 
    default_config=False
)

ssa.setup()

# Scale up (perturb) the voltage states of both p0 and q0 by 10%. 
ssa.PQ.set(
            src='p0', 
            idx= ssa.PQ.idx.v, 
            attr='v',
            value=1.1 * ssa.PQ.p0.v
        )

ssa.PQ.set(
            src='q0', 
            idx= ssa.PQ.idx.v, 
            attr='v',
            value=1.1 * ssa.PQ.q0.v
        )

ssa.PFlow.run()

print("PQ.p0.v After Scaling:", ssa.PQ.p0.v)
print("PQ.q0.v After Scaling:", ssa.PQ.q0.v)


# Load the IEEE 39-bus case, scale the active and reactive power demands of all PQ loads by 10%, then run a power-flow simulation and display the updated values.
