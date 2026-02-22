# required_dependencies: pandas,pypower
import pandas as pd
from pypower.api import runpf, case9

# Load a standard case (e.g., case9) from PYPOWER
# This is a simple 9-bus system
ppc = case9()

# Run power flow calculations
# 'case9' is the case we are using which represents data for the system
success, results = runpf(ppc)

# Check if the power flow calculation was successful
if success:
    print("Power flow calculation was successful!")
    # Extracting bus data from results
    bus_data = pd.DataFrame(results['bus'], columns=['Bus', 'Vm', 'Va', 'Pd', 'Qd'])
    print(bus_data)
else:
    print("Power flow calculation failed.")