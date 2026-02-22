# required_dependencies: pandas, numpy, matplotlib, andes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from andes import Andes

# Initialize Andes
andes = Andes()

# Load a sample power system case (You may replace this with your own case)
# Here we assume you have a case file. You can load various power flow cases.
case_file = "path_to_your_case_file" # replace with actual path

# Load the case
andes.load_case(case_file)

# Run the power flow analysis
results = andes.run_power_flow()

# Display results
print("Power Flow Results:")
print(results)

# Optionally, you can visualize the results if needed
# For example if you want to plot bus voltages:
bus_voltages = results['bus_voltages'] # assuming results have bus_voltage data

plt.figure(figsize=(10, 6))
plt.bar(range(len(bus_voltages)), bus_voltages, color='blue')
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')
plt.title('Bus Voltages After Power Flow Analysis')
plt.show()