import andes

ssa = andes.load(
    andes.get_case('ieee39/ieee39.xlsx'),
    setup=False,     # If setup=False, need to call ss.setup() before running the simulation
    no_output=True, 
    default_config=False
)

print("PQ Table of Case:", ssa.PQ.as_df())

# Load the IEEE 39-bus case and display the PQ load table as a DataFrame.