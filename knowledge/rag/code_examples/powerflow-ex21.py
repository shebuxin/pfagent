import andes


ssa = andes.load(
    andes.get_case('ei/EI_33.xlsx')
)

ssa.PFlow.run()


# Read per‐unit outputs
p0_slack = ssa.Slack.p0.v  # the slack‐bus gen

print("Initial Power for All Slack Bus Components:", p0_slack)


# Print the initial power of all slack bus components.