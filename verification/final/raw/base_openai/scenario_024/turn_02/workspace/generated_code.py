
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