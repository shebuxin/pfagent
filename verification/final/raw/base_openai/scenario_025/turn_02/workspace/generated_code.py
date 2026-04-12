
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

# required_dependencies: pandas, numpy, andes
import pandas as pd
import numpy as np
import andes

# Load the case
case = "verify_ieee14_025.xlsx"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.020
ssa.PV.vset.v[0] = 1.020  # Assuming the first PV is at index 0

# Rerun power flow analysis
andes.run_powerflow(ssa)

# Get the affected PV bus voltage and the number of buses above 1.025 p.u.
pv_bus = ssa.PV.bus.v[0]  # Bus number of the first PV
pv_voltage = ssa.PV.v.v[0]  # Voltage at that bus

# Count how many buses have voltages above 1.025 p.u.
above_threshold_count = np.sum(ssa.Bus.v.v > 1.025)

# Prepare results
RESULT_JSON = {
    "pv_bus": int(pv_bus),
    "pv_setpoint": round(1.020, 6),  # Setpoint value rounded to 6 decimals
    "pv_voltage": round(float(pv_voltage), 6),
    "selected_count": int(above_threshold_count)
}

# Print the result in the required format
print(f"RESULT_JSON={RESULT_JSON}")