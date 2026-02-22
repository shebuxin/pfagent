import ANDES

# Load the IEEE 39-bus case and run an AC power flow
ssa = ANDES.load_case('ieee39.xlsx')
ssa.PFlow.run()

# Print all bus voltage angles (radians)
print("Bus Angles (Radians):")
print(ssa.Bus.a.v)

# Print all bus voltages (per unit)
print("Bus Voltages (p.u.):")
print(ssa.Bus.v.v)