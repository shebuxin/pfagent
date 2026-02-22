import andes
import numpy as np

ssa = andes.load(
    andes.get_case('ieee39/ieee39.xlsx'),
    setup=True,     # If setup=False, need to call ss.setup() before running the simulation
    no_output=True, 
    default_config=False
)

# Scale up (perturb) the voltage states of both p0 and q0 by 10%. 
ssa.PQ.set(
            src='p0', 
            idx= ssa.PQ.idx.v, 
            attr='v',
            value= 1.1 * ssa.PQ.p0.v
        )

ssa.PQ.set(
            src='q0', 
            idx= ssa.PQ.idx.v, 
            attr='v',
            value= 1.1 * ssa.PQ.q0.v
        )

# Change specific load based on idx
# Note: check the case (excel) file to get the correct idx
ssa.PQ.set(
            src='p0', 
            idx= ['PQ_1', 'PQ_10'], 
            attr='v',
            value= np.array([8, 4])
        )

ssa.PQ.set(
            src='q0', 
            idx= ['PQ_1', 'PQ_10'], 
            attr='v',
            value= np.array([2, 1])
        )

ssa.PFlow.run()

print("PQ.p0.v After Change:", ssa.PQ.p0.v)
print("PQ.q0.v After Change:", ssa.PQ.q0.v)


# Load the IEEE 39-bus case, increase all PQ load active and reactive powers by 10%, modify specific loads by index, then run a power-flow simulation and display the updated values.