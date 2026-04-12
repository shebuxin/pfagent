
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

# Run power flow
result = andes.run_power_flow(ssa)

# Find max and min voltage buses
max_voltage_idx = result.V.idx.v[np.argmax(result.V.mag.v)]
min_voltage_idx = result.V.idx.v[np.argmin(result.V.mag.v)]
max_voltage = round(result.V.mag.v[max_voltage_idx], 6)
min_voltage = round(result.V.mag.v[min_voltage_idx], 6)

# Prepare result JSON
result_json = {
    "max_bus": int(max_voltage_idx),
    "max_voltage": float(max_voltage),
    "min_bus": int(min_voltage_idx),
    "min_voltage": float(min_voltage)
}

# Print result
print("RESULT_JSON=" + json.dumps(result_json))