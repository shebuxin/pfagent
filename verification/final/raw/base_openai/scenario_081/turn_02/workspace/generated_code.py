
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

# required_dependencies: pandas, numpy
import os
import andes
import numpy as np

# Load the case from the uploaded Excel file
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_081.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.010
if ssa.PV.idx.v.size > 0:
    first_pv_index = ssa.PV.idx.v[0]
    ssa.PV.set(first_pv_index, "Vtarget", 1.010)

# Rerun power flow
andes.run_power_flow(ssa)

# Retrieve the affected PV bus voltage
affected_pv_voltage = ssa.PV.v[0]  # First PV's voltage

# Count how many buses are above 1.010 p.u.
buses_above_target = np.sum(ssa.bus.v > 1.010)

# Prepare the RESULT_JSON
result_json = {
    "pv_bus": int(ssa.PV.bus.v[0]),
    "pv_setpoint": round(1.010, 6),
    "pv_voltage": round(float(affected_pv_voltage), 6),
    "selected_count": int(buses_above_target)
}

# Print the result in the specified format
print(f"RESULT_JSON={result_json}")