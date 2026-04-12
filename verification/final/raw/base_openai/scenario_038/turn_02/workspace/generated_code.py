
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
import numpy as np

# Load IEEE 39 case
case = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True)

# Set the first PV voltage target to 1.010
case.PV.set_targets([1.010])

# Rerun power flow
case.run()

# Inspect results
pv_bus = int(case.PV.bus[0])
pv_setpoint = float(case.PV.target.v[0])
pv_voltage = float(case.V[np.where(case.bus == pv_bus)])

# Count how many buses are above 1.030 p.u.
selected_count = int(np.sum(case.V > 1.030))

# Prepare result JSON
result_json = {
    "pv_bus": pv_bus,
    "pv_setpoint": round(pv_setpoint, 6),
    "pv_voltage": round(pv_voltage, 6),
    "selected_count": selected_count
}

# Print the result
print("RESULT_JSON=", result_json)