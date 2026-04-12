
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

# Load the uploaded case
case = "verify_kundur_080.xlsx"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.000
pv_index = ssa.PV.idx.v[0]  # Get the index of the first PV
ssa.PV.set(pv_index, 'Vtarget', 1.000)

# Rerun power flow
result = andes.run(ssa)

# Get the affected PV bus voltage
affected_pv_voltage = round(ssa.PV.v[vv.pv_index], 6)

# Count the number of buses above 1.000 p.u.
above_target_count = (ssa.bus.v > 1.000).sum()

# Prepare JSON result
RESULT_JSON = {
    "pv_bus": int(ssa.PV.bus.v[pv_index]),
    "pv_setpoint": 1.000,
    "pv_voltage": float(affected_pv_voltage),
    "selected_count": int(above_target_count),
}

print("RESULT_JSON=", RESULT_JSON)