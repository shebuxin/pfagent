
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

case = andes.load(andes.get_case("kundur/kundur_full.xlsx"), setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.000
first_pv_index = 0  # Assuming the first PV is at index 0
case.PV.set(first_pv_index, V_target=1.000)

# Rerun power flow
result = andes.run_power_flow(case)

# Collecting results
pv_bus = case.PV.bus.v[first_pv_index]
pv_setpoint = 1.000
pv_voltage = round(result.V.mag.v[case.PV.bus.v[first_pv_index] - 1], 6)  # Adjusting for 0-based index
selected_count = sum(result.V.mag.v > 1.000)

RESULT_JSON = {
    "pv_bus": int(pv_bus),
    "pv_setpoint": float(pv_setpoint),
    "pv_voltage": float(pv_voltage),
    "selected_count": int(selected_count)
}

print("RESULT_JSON=", RESULT_JSON)