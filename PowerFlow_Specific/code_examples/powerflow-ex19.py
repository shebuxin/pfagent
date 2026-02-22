import andes
import numpy as np


ssa = andes.load(
    andes.get_case('5bus/pjm5bus.xlsx')
)

ssa.PFlow.run()


# Read per‐unit outputs
v0_pv = ssa.PV.v0.v  # all PV‐type gens
v0_slack = ssa.Slack.v0.v  # the slack‐bus gen

# Combine into one array
v0_all_gen = np.concatenate([v0_pv, v0_slack])

print("Initial Voltage for All Generators:", v0_all_gen)


# Print the initial voltage of all generators.