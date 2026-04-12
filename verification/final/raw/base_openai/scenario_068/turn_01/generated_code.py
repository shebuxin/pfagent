# required_dependencies: pandas
import andes
import pandas as pd

# Load the Kundur full case
case = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run the power flow
andes.run("powerflow", ssa)

# Get the bus voltages
bus_voltages = pd.DataFrame({
    'bus_id': ssa.Bus.idx.v,
    'voltage': ssa.Bus.vmag.v
})

# Find the two lowest voltage buses
lowest_buses = bus_voltages.nsmallest(2, 'voltage').sort_values('voltage')

# Prepare the result
RESULT_JSON = {
    "selected_bus_ids": lowest_buses['bus_id'].tolist(),
    "selected_voltages": [round(v, 6) for v in lowest_buses['voltage'].tolist()]
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")