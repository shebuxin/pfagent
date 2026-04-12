
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

# required_dependencies: pandas
import os
import andes
import json
import numpy as np

# Load the case
script_dir = os.getcwd()
case_file = os.path.join(script_dir, "verify_ieee14_020.xlsx")
ssa = andes.load(case_file, setup=True, no_output=True, log=False)

# Set slack bus voltage target
slack_setpoint = 1.025
slack_bus_idx = ssa.SlackBus.idx.v[0]  # Assuming there's only one slack bus
ssa.SlackBus.set(bus=slack_bus_idx, v_target=slack_setpoint)

# Add a new PQ load at bus 5
added_load_idx = "PQ_VERIFY_020_B"
p0 = 0.014
q0 = 0.009
bus_id_5 = 5  # Assuming bus number 5 is correct
ssa.PQ.add(idx=added_load_idx, bus=bus_id_5, p0=p0, q0=q0)

# Run power flow
andes.run_power_flow(ssa)

# Get the bus voltages
bus_voltages = ssa.Bus.v
bus_ids = ssa.Bus.idx

# Find max and min voltage buses
max_voltage_index = np.argmax(bus_voltages)
min_voltage_index = np.argmin(bus_voltages)

max_bus = bus_ids[max_voltage_index]
max_voltage = round(bus_voltages[max_voltage_index], 6)
min_bus = bus_ids[min_voltage_index]
min_voltage = round(bus_voltages[min_voltage_index], 6)

# Count total number of PQ loads
total_pq_count = len(ssa.PQ.idx.v)

# Prepare the result as required
result_json = {
    "added_load_idx": added_load_idx,
    "max_bus": max_bus,
    "max_voltage": max_voltage,
    "min_bus": min_bus,
    "min_voltage": min_voltage,
    "total_pq_count": total_pq_count
}

# Print the result in the specified format
print("RESULT_JSON=" + json.dumps(result_json))