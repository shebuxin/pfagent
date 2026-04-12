# required_dependencies: pandas, numpy
import pandas as pd
import numpy as np
import andes

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run the power flow analysis
andes.run(ssa)

# Define the voltage threshold
threshold = 1.030

# Count all buses above the threshold
above_threshold = ssa.Bus.voltage.v > threshold
selected_bus_ids = ssa.Bus.idx.v[above_threshold].tolist()
selected_count = len(selected_bus_ids)

# Find the two lowest-voltage buses
bus_voltages = pd.DataFrame({
    'bus_id': ssa.Bus.idx.v,
    'voltage': ssa.Bus.voltage.v
})

lowest_buses = bus_voltages.nsmallest(2, 'voltage').sort_values(by='voltage')
lowest_bus_ids = lowest_buses['bus_id'].tolist()
lowest_voltages = lowest_buses['voltage'].round(6).tolist()

# Construct the result JSON
result_json = {
    'threshold': round(threshold, 6),
    'selected_bus_ids': selected_bus_ids,
    'selected_count': selected_count,
    'lowest_bus_ids': lowest_bus_ids,
    'lowest_voltages': lowest_voltages
}

# Print the result
print(f"RESULT_JSON={result_json}")