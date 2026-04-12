# required_dependencies: andes
import andes

# Load the Kundur built-in case
case = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.010
pv_idx = ssa.PV.idx.v[0]
ssa.PV.set(pv_idx, target_voltage=1.010)

# Rerun power flow
andes.run(ssa, log=False)

# Retrieve the affected PV bus voltage
pv_bus = int(ssa.PV.bus.v[0])
pv_setpoint = 1.010
pv_voltage = round(float(ssa.Bus.voltage.v[pv_bus - 1]), 6)  # Adjust for 0-based indexing

# Count how many buses are above the setpoint
selected_count = (ssa.Bus.voltage.v > pv_setpoint).sum()

# Output results in the required JSON format
RESULT_JSON = {
    "pv_bus": pv_bus,
    "pv_setpoint": pv_setpoint,
    "pv_voltage": pv_voltage,
    "selected_count": selected_count
}

print(f"RESULT_JSON={RESULT_JSON}")