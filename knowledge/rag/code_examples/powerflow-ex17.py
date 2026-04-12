import andes
import numpy as np


ssa = andes.load(
    andes.get_case('ieee39/ieee39.xlsx')
)

ssa.PFlow.run()


# Read per‐unit outputs
p_pv = ssa.PV.p.v  # all PV‐type gens
p_slack = ssa.Slack.p.v  # the slack‐bus gen

# Combine into one array
p_all_pu = np.concatenate([p_pv, p_slack])

# Convert to MW
p_all_mw = p_all_pu * ssa.config.mva

# Find min
idx_max = np.argmin(p_all_mw)

# Map back to names
names_all = ssa.PV.name.v + ssa.Slack.name.v

print(f"Min Generator: {names_all[idx_max]} at {p_all_mw[idx_max]} MW")


# Load the IEEE 39-bus case, run a power-flow simulation, and find the generator with the lowest active power output, displaying its name and output in MW.