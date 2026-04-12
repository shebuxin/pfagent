# required_dependencies: andes
import andes

# Load the uploaded case
case = "verify_kundur_080.xlsx"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.000
pv_index = ssa.PV.idx.v[0]  # Get the index of the first PV
ssa.PV.set(pv_index, 'Vtarget', 1.000)

# Rerun power flow
result = andes.run(ssa)

# Get the affected PV bus voltage
affected_pv_voltage = round(ssa.PV.v[vv.pv_index], 6)

# Count the number of buses above 1.000 p.u.
above_target_count = (ssa.bus.v > 1.000).sum()

# Prepare JSON result
RESULT_JSON = {
    "pv_bus": int(ssa.PV.bus.v[pv_index]),
    "pv_setpoint": 1.000,
    "pv_voltage": float(affected_pv_voltage),
    "selected_count": int(above_target_count),
}

print("RESULT_JSON=", RESULT_JSON)