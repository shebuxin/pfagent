
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

# required_dependencies: pandas, matplotlib
import andes
import pandas as pd
import matplotlib.pyplot as plt

# Load the built-in PJM 5-bus case
case = andes.load(andes.get_case("5bus/pjm5bus.json"), setup=True, no_output=True)

# Scale every PQ load by 1.040
scale_factor = 1.040
case.PQ.load.v *= scale_factor

# Add a new PQ load at bus 2
new_load_idx = 'PQ_VERIFY_086_E'
case.PQ.add(idx=new_load_idx, bus=2, p0=0.014, q0=0.008)

# Rerun power flow
case.run_power_flow()

# Get bus voltages
bus_voltages = case.Bus.v
bus_indices = case.Bus.idx.v

# Identify max and min voltages
max_voltage = bus_voltages.max()
min_voltage = bus_voltages.min()
max_bus = bus_indices[bus_voltages.argmax()]
min_bus = bus_indices[bus_voltages.argmin()]

# Save a line plot of bus voltages
plt.figure()
plt.plot(bus_indices, bus_voltages, marker='o', linestyle='-')
plt.title('Bus Voltages')
plt.xlabel('Bus Index')
plt.ylabel('Voltage (PU)')
plt.grid()
plt.savefig('scenario_086_turn3_voltage.png')

# Prepare the RESULT_JSON output
RESULT_JSON = {
    "added_load_idx": new_load_idx,
    "max_bus": int(max_bus),
    "max_voltage": round(max_voltage, 6),
    "min_bus": int(min_bus),
    "min_voltage": round(min_voltage, 6),
    "plot_file": 'scenario_086_turn3_voltage.png'
}

# Print the RESULT_JSON
print(f"RESULT_JSON={RESULT_JSON}")