import andes
import numpy as np

ssa = andes.load(
    andes.get_case('ieee39/ieee39.xlsx'),
    setup=False,     # If setup=False, need to call ss.setup() before running the simulation
    no_output=True, 
    default_config=False
)

ssa.setup()

# change power reference of PV/Slack bus 

# PV bus
# Note: check the case (excel) file to get the correct idx
# If the idx is number, then use idx= [1, 2] instead of idx= ['1', '2']
ssa.PV.set(
            src='p0', 
            idx= [1, 2], 
            attr='v',
            value= np.array([5, 7])
        )

ssa.PV.set(
            src='q0', 
            idx= [1, 2], 
            attr='v',
            value= np.array([2, 2.5])
        )

ssa.PFlow.run()

print("PV.p0.v After Change:", ssa.PV.p0.v)
print("PV.q0.v After Change:", ssa.PV.q0.v)


# Load the IEEE 39-bus case, modify the active and reactive power setpoints for selected PV buses, then run a power-flow simulation and display the updated values.