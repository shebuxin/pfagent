# Required dependencies: andes, pandas

import andes
import pandas as pd

# Load the IEEE 39-bus case from Excel
ssa = andes.load(
    pd.ExcelFile(
        "ieee39.xlsx"
    )
)

# Run the power flow simulation
ssa.PFlow.run()

# Print all bus voltage angles (radians)
print("Bus Angles (Radians):")
print(ssa.Bus.a.v)

# Print all bus voltages (per unit)
print("Bus Voltages (p.u.):")
print(ssa.Bus.v.v)