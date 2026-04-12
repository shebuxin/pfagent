
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

# Load the built-in Kundur case
case = andes.get_case("kundur/kundur_aw.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run_powerflow(ssa)

# Report maximum and minimum voltages
max_voltage_idx = ssa.Bus.v.argmax()
min_voltage_idx = ssa.Bus.v.argmin()

# Prepare the result
result_json = {
    "max_bus": int(ssa.Bus.idx.v[max_voltage_idx]),
    "max_voltage": round(float(ssa.Bus.v[max_voltage_idx]), 6),
    "min_bus": int(ssa.Bus.idx.v[min_voltage_idx]),
    "min_voltage": round(float(ssa.Bus.v[min_voltage_idx]), 6),
}

# Print result
print(f"RESULT_JSON={result_json}")