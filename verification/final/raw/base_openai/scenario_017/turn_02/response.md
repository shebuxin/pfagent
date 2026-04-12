Here's the updated Python script that incorporates your specifications, including adding a new PQ load and reporting the required information:

```python
# required_dependencies: andes
import andes
import os
import json

# Load the case from the uploaded file
script_dir = os.getcwd()
case_filename = "verify_ieee14_017.xlsx"
case_path = os.path.join(script_dir, case_filename)

# Load the case setup
ssa = andes.load(case_path, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 9
new_load_idx = 'PQ_VERIFY_017_A'
p0 = 0.016
q0 = 0.01
bus_idx = np.where(ssa.Bus.idx.v == 9)[0][0]  # Find the index of bus 9
ssa.PQ.add(idx=new_load_idx, bus=bus_idx, p0=p0, q0=q0)

# Run power flow analysis with the added load
andes.run_power_flow(ssa)

# Define the voltage threshold
threshold = 1.010

# Get all buses below the threshold
voltages = ssa.Bus.V.v
selected_bus_indices = np.where(voltages < threshold)[0]  # Get indices of buses below threshold
selected_bus_ids = ssa.Bus.idx.v[selected_bus_indices].tolist()

# Find the minimum voltage bus
min_voltage = voltages.min()
min_bus_idx = ssa.Bus.idx.v[np.where(voltages == min_voltage)[0][0]]

# Prepare the result JSON
result_json = {
    "added_load_idx": new_load_idx,
    "added_load_bus": bus_idx,
    "threshold": threshold,
    "selected_bus_ids": sorted(selected_bus_ids),
    "selected_count": len(selected_bus_ids),
    "min_bus": int(min_bus_idx),
    "min_voltage": round(min_voltage, 6),
}

# Print the result JSON
print(f"RESULT_JSON={json.dumps(result_json)}")
```

Make sure you have `andes` installed and configured correctly in your environment to run this script. It will add the specified load and provide the desired output as requested.