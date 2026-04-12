
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

# required_dependencies: pandas
import andes
import os

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.005
pv_idx = ssa.PV.idx.v[0]
ssa.PV.vset.v[pv_idx] = 1.005

# Rerun the power flow analysis
andes.powerflow(ssa)

# Get the updated voltage of the first PV bus
pv_voltage = ssa.PV.v.v[pv_idx]
pv_bus = ssa.PV.bus.v[pv_idx]

# Count the number of buses above 1.020 p.u.
above_threshold_count = (ssa.Bus.v.v > 1.020).sum()

# Prepare the JSON output
RESULT_JSON = {
    "pv_bus": int(pv_bus),
    "pv_setpoint": round(1.005, 6),
    "pv_voltage": round(float(pv_voltage), 6),
    "selected_count": int(above_threshold_count)
}

# Print the result
print("RESULT_JSON=" + str(RESULT_JSON))