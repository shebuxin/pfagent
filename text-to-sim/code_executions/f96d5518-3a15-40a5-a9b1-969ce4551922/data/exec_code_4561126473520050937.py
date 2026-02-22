import andes

ssa = ANDES.load_case('ieee39/ieee39.xlsx')  # load the standard IEEE 39-bus case
ssa.PFlow.run()                         # execute a full nonlinear AC power flow

print("Bus Angles (radians):")
print(ssa.Bus.a.v)                     # display bus voltage angles in radians

print("\nBus Voltages (p.u.):")
print(ssa.Bus.v.v)                     # display bus voltage magnitudes in per unit