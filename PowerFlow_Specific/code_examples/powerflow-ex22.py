import andes


ssa = andes.load(
    andes.get_case('ieee14/ieee14_full.xlsx')
)

ssa.PFlow.run()


bus_names = ssa.Bus.name.v

print("Bus Names:", bus_names)


# Print all Bus names.