
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

# required_dependencies: andes, matplotlib
import andes
import matplotlib.pyplot as plt

case = andes.load(andes.get_case("kundur/kundur_full.xlsx"), setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.000
first_pv_index = 0  # Assuming the first PV is at index 0
case.PV.set(first_pv_index, V_target=1.000)

# Scale every PQ load by 1.040
scale_factor = 1.040
case.PQ.set_all(P=case.PQ.P.v * scale_factor, Q=case.PQ.Q.v * scale_factor)

# Rerun power flow
result = andes.run_power_flow(case)

# Collecting results for JSON output
min_bus_index = result.V.idx.v[np.argmin(result.V.mag.v)]
max_bus_index = result.V.idx.v[np.argmax(result.V.mag.v)]
min_voltage = round(np.min(result.V.mag.v), 6)
max_voltage = round(np.max(result.V.mag.v), 6)

RESULT_JSON = {
    "scale_factor": float(scale_factor),
    "min_bus": int(min_bus_index),
    "min_voltage": float(min_voltage),
    "max_bus": int(max_bus_index),
    "max_voltage": float(max_voltage),
    "plot_file": "scenario_066_turn3_bar.png"
}

# Create bar chart of bus voltages
plt.bar(result.V.idx.v, result.V.mag.v)
plt.xlabel('Bus Index')
plt.ylabel('Voltage (p.u.)')
plt.title('Bus Voltages After Power Flow')
plt.savefig(RESULT_JSON["plot_file"])
plt.close()

print("RESULT_JSON=", RESULT_JSON)