import andes
import numpy as np

ssa = andes.load(
    andes.get_case('ieee39/ieee39.xlsx'),
    setup=True,     # If setup=False, need to call ss.setup() before running the simulation
    no_output=True, 
    default_config=False
)

# Slack bus
# Note: check the case (excel) file to get the correct src and idx
ssa.Slack.set(
            src='v0', 
            idx= 10, 
            attr='v',
            value= np.array([1])  # value= 1  works too
        )

ssa.Slack.set(
            src='a0', 
            idx= 10, 
            attr='v',
            value= np.array([0])  # value= 0  works too
        )

ssa.PFlow.run()

print("Slack.v0.v After Update", ssa.Slack.v0.v)
print("Slack.a0.v After Update", ssa.Slack.a0.v)


# Load the IEEE 39-bus case, update the voltage magnitude and angle setpoints for the Slack bus, then run a power-flow simulation and display the updated values.