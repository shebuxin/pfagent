import andes


ssa = andes.load(
    andes.get_case('kundur/kundur_full.xlsx')
)

ssa.PFlow.run()


print("PV Bus Active Power", ssa.PV.p.v)
print("PV Bus Reactive Power", ssa.PV.q.v)
print("Slack Bus Active Power", ssa.Slack.p.v)
print("Slack Bus Reactive Power", ssa.Slack.q.v)


# Print active and reactive power of all generators.