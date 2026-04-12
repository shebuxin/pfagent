# required_dependencies: andes
import andes

# Load the built-in Kundur case
case = andes.get_case("kundur/kundur_aw.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 0.990
ssa.PV.set(0, target_voltage=0.990)

# Rerun power flow
andes.run_powerflow(ssa)

# Get the affected PV bus voltage
pv_bus = int(ssa.PV.bus.v[0])
pv_setpoint = 0.990
pv_voltage = round(float(ssa.Bus.v[ssa.PV.bus.v[0]]), 6)

# Count how many buses are above 0.990 p.u.
selected_count = (ssa.Bus.v > 0.990).sum()

# Prepare the result
result_json = {
    "pv_bus": pv_bus,
    "pv_setpoint": pv_setpoint,
    "pv_voltage": pv_voltage,
    "selected_count": int(selected_count),
}

# Print result
print(f"RESULT_JSON={result_json}")