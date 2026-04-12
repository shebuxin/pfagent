import andes

ssa = andes.load(
    andes.get_case('ieee39/ieee39.xlsx'),
    setup=True,
    no_output=True,
)

# call the power flow calculation
ssa.PFlow.run()

print("PV Bus Active Power", ssa.PV.p.v)
print("PV Bus Reactive Power", ssa.PV.q.v)
print("Slack Bus Active Power", ssa.Slack.p.v)
print("Slack Bus Reactive Power", ssa.Slack.q.v)
print("Bus Voltages", ssa.Bus.v.v)
print("Bus Angles", ssa.Bus.a.v)


# Load the IEEE 39-bus case, run a power-flow simulation, and display key results for the PV buses, Slack bus, and all system buses.
