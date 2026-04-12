# required_dependencies: andes
import os
import andes

# Load the case
case = os.path.join(os.getcwd(), "verify_ieee14_024.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target
first_pv_idx = ssa.PV.idx.v[0]
ssa.PV.set(first_pv_idx, Vtarget=1.015)

# Run power flow again
andes.run(ssa, solver='default', log=True)

# Analyze the affected PV voltage and count buses above 1.020 p.u.
pv_bus = ssa.PV.bus.v[first_pv_idx]
pv_voltage = round(ssa.bus.v[first_pv], 6)
selected_count = (ssa.bus.v > 1.020).sum()

# Prepare results
RESULT_JSON = {
    "pv_bus": int(pv_bus),
    "pv_setpoint": float(1.015),
    "pv_voltage": float(pv_voltage),
    "selected_count": int(selected_count)
}

# Print result in specified JSON format
print(f"RESULT_JSON={RESULT_JSON}")