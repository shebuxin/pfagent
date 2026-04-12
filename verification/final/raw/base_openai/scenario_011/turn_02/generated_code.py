# required_dependencies: andes
import andes

# Load the built-in IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.020 p.u.
pv_indices = ssa.PV.idx.v
if len(pv_indices) > 0:
    ssa.PV.Vset.v[0] = 1.020

# Rerun the power flow analysis
andes.run_power_flow(ssa)

# Get the affected PV bus voltage and check for buses above 1.025 p.u.
pv_bus = ssa.PV.bus.v[0]
pv_setpoint = round(ssa.PV.Vset.v[0], 6)
pv_voltage = round(ssa.Vmag.v[pv_bus], 6)
selected_count = (ssa.Vmag.v > 1.025).sum()

# Prepare the RESULT_JSON
RESULT_JSON = {
    "pv_bus": pv_bus,
    "pv_setpoint": pv_setpoint,
    "pv_voltage": pv_voltage,
    "selected_count": selected_count
}

# Print the result in the required format
print(f"RESULT_JSON={RESULT_JSON}")