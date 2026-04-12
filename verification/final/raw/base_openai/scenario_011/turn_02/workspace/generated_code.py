
try:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _verification_output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(_verification_output_dir, exist_ok=True)
    _verification_existing = [
        name for name in os.listdir(_verification_output_dir)
        if name.startswith("plot_") and name.endswith(".png")
    ]
    _verification_plot_counter = len(_verification_existing)

    def _verification_safe_show(*args, **kwargs):
        global _verification_plot_counter
        saved_paths = []
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            _verification_plot_counter += 1
            plot_path = os.path.join(_verification_output_dir, f"plot_{_verification_plot_counter}.png")
            fig.savefig(plot_path, bbox_inches="tight")
            saved_paths.append(plot_path)
        if saved_paths:
            print("Saved plot(s):")
            for path in saved_paths:
                print(f"- {path}")
        plt.close("all")

    plt.show = _verification_safe_show
except Exception:
    pass

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