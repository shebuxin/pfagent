
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

# required_dependencies: pandas, numpy, andes, matplotlib
import pandas as pd
import numpy as np
import andes
import matplotlib.pyplot as plt

# Load the case
case = "verify_ieee14_025.xlsx"
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target to 1.020
ssa.PV.vset.v[0] = 1.020

# Scale every PQ load by 1.050
ssa.PQ.P.v *= 1.050
ssa.PQ.Q.v *= 1.050

# Rerun power flow analysis
andes.run_powerflow(ssa)

# Get bus voltages
voltages = ssa.Bus.v.v

# Calculate min and max voltage details
min_voltage = np.min(voltages)
max_voltage = np.max(voltages)
min_bus = np.where(voltages == min_voltage)[0][0] + 1  # Convert index to bus number (1-based)
max_bus = np.where(voltages == max_voltage)[0][0] + 1  # Convert index to bus number (1-based)

# Create a bar chart of the bus voltages
plt.bar(range(1, len(voltages) + 1), voltages)  # Buses are 1-based
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')
plt.title('Bus Voltages')
plt.ylim(0.9, 1.1)  # Set limits to focus on relevant voltage range
plot_file = 'scenario_025_turn3_bar.png'
plt.savefig(plot_file)
plt.close()

# Prepare results
RESULT_JSON = {
    "scale_factor": 1.050,
    "min_bus": int(min_bus),
    "min_voltage": round(float(min_voltage), 6),
    "max_bus": int(max_bus),
    "max_voltage": round(float(max_voltage), 6),
    "plot_file": plot_file
}

# Print the result in the required format
print(f"RESULT_JSON={RESULT_JSON}")