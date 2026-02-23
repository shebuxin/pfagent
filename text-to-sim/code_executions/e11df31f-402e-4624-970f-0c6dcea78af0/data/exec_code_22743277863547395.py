import andes

ssa = andes.load(
    andes.get_case('ieee39/ieee39.xlsx'),
    setup=False, no_output=True, preserve_case=True
)


ssa.setup()
ssa.PFlow.run()