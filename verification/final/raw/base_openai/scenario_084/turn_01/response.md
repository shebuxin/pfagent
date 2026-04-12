```python
# required_dependencies: andes,pandas
import andes
import pandas as pd
import os

# Load the uploaded case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_084.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run()

# Retrieve bus voltages
bus_voltages = pd.DataFrame({
    'bus_id': ssa.bus.idx.v,
    'voltage': ssa.bus.vmag.v
})

# Find the 4 lowest-voltage buses
lowest_voltages = bus_voltages.nsmallest(4, 'voltage')

# Create the result JSON
result_json = {
    'selected_bus_ids': list(lowest_voltages['bus_id']),
    'selected_voltages': [round(v, 6) for v in list(lowest_voltages['voltage'])]
}

# Print the result
print(f'RESULT_JSON={result_json}')
```