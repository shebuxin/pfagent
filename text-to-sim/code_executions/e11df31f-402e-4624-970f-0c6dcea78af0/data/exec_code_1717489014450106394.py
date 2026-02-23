import andes

ssa = andes.load(
    andes.get_case('ieee39/ieee39.xlsx'),
    setup=False, no_output=True, preserve_case=True
)


ssa.setup()
ssa.PFlow.run()

print("Bus Voltages:", ssa.Bus.v.v)
print("Bus Angles:", ssa.Bus.a.v)
print("PQ Bus Active Power:", ssa.PQ.p0.v)
print("PQ Bus Reactive Power:", ssa.PQ.q0.v)
print("PV Bus Active Power:", ssa.PV.p0.v)
print("PV Bus Reactive Power:", ssa.PV.q0.v)