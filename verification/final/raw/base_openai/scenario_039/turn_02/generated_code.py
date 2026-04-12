# required_dependencies: andes
import andes

# Load the IEEE 39 case
case = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.015
first_pv_idx = case.PV.idx.v[0]
case.PV.voltage_target.set(first_pv_idx, 1.015)

# Rerun power flow analysis
case.solve_power_flow()

# Extract the affected PV bus voltage and count how many buses are above 1.040 p.u.
pv_bus = int(case.PV.bus.v[first_pv_idx])
pv_setpoint = round(float(case.PV.voltage_target.v[first_pv_idx]), 6)
pv_voltage = round(float(case.Bus.v[pv_bus]), 6)
selected_count = (case.Bus.v > 1.040).sum()

# Create the JSON result
RESULT_JSON = {
    "pv_bus": pv_bus,
    "pv_setpoint": pv_setpoint,
    "pv_voltage": pv_voltage,
    "selected_count": int(selected_count)
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")