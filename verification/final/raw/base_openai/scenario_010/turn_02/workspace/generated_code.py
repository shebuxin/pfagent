
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

# required_dependencies: andes,json
import andes
import json

# Load the IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.015
pv_bus_idx = ssa.PV.idx.v[0]
ssa.PV.vset.v[pv_bus_idx] = 1.015

# Rerun power flow
result = andes.run_power_flow(ssa)

# Get the affected PV bus voltage
pv_voltage = round(result.V.mag.v[pv_bus_idx], 6)

# Count how many buses are above 1.020 p.u.
selected_count = (result.V.mag.v > 1.020).sum()

# Prepare result JSON
result_json = {
    "pv_bus": int(pv_bus_idx),
    "pv_setpoint": float(1.015),
    "pv_voltage": float(pv_voltage),
    "selected_count": int(selected_count)
}

# Print result
print("RESULT_JSON=" + json.dumps(result_json))