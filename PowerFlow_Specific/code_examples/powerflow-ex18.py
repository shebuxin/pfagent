import andes
import numpy as np


ssa = andes.load(
    andes.get_case('ieee39/ieee39.xlsx')
)

ssa.PFlow.run()


# Read per‐unit outputs
q_pv = ssa.PV.q.v  # all PV‐type gens
q_slack = ssa.Slack.q.v  # the slack‐bus gen

# Combine into one array
q_all_pu = np.concatenate([q_pv, q_slack])

# Convert to Mvar
q_all_mvar = q_all_pu * ssa.config.mva

# Find min
idx_max = np.argmin(q_all_mvar)

# Map back to names
names_all = ssa.PV.name.v + ssa.Slack.name.v

print(f"Min Generator: {names_all[idx_max]} at {q_all_mvar[idx_max]} Mvar")


# Load the IEEE 39-bus case, run a power-flow simulation, and find the generator with the lowest reactive power output, displaying its name and output in Mvar.