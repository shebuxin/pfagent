import andes


ssa = andes.load(
    andes.get_case('GBnetwork/GBnetwork.xlsx')
)

ssa.PFlow.run()


# Read per‐unit outputs
v0_pq = ssa.PQ.v0.v  # all PQ‐type loads

print("Initial Voltage for All Loads:", v0_pq)


# Print the initial voltage of all loads.