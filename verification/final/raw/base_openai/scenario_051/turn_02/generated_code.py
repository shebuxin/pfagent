# required_dependencies: andes
import andes

# Load the case
case = "verify_ieee39_051.xlsx"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.005
first_pv_idx = ssa.PV.idx.v[0]
ssa.PV.v_target.set(first_pv_idx, 1.005)

# Rerun power flow
andes.run_power_flow(ssa)

# Get the voltage at the affected PV bus
pv_bus = ssa.PV.bus.v[first_pv_idx]
pv_voltage = round(ssa.Bus.v[pv_bus], 6)

# Count how many buses are above 1.020 p.u.
selected_count = (ssa.Bus.v > 1.020).sum()

# Prepare the result
RESULT_JSON = {
    "pv_bus": int(pv_bus),
    "pv_setpoint": 1.005,
    "pv_voltage": float(pv_voltage),
    "selected_count": int(selected_count),
}

# Print the result in the required format
print(f'RESULT_JSON={RESULT_JSON}')