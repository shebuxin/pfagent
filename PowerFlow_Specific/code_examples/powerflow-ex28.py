import andes
import numpy as np
import os

script_dir = os.getcwd()
case = os.path.join(script_dir, 'EI_33.xlsx')

ssa = andes.load(
    case
)

ssa.PFlow.run()

# Extract active and reactive power for all Slack generators
slack_p = ssa.Slack.p.v  # real power (P)
slack_q = ssa.Slack.q.v  # reactive power (Q)
slack_sn = ssa.Slack.Sn.v  # apparent power (Sn)
slack_ids = ssa.Slack.idx.v

# Compute apparent power S = sqrt(P^2 + Q^2)
s_apparent = np.sqrt(slack_p**2 + slack_q**2)

# Print results
for i, slack_generator in enumerate(slack_ids):
    print(f"Generator {slack_generator}: P = {slack_p[i]:.4f}, Q = {slack_q[i]:.4f}, |S| = {s_apparent[i]:.4f}, Sn = {slack_sn[i]:.4f}")
    print(f"Generator {slack_generator}: Remaining Capacity = {slack_sn[i] - s_apparent[i]}")


# Remaining generation capability for slack bus generators
# Units = MVA?