
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

# required_dependencies: numpy, pandas, matplotlib, andes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import andes

# Load the built-in IEEE 39 case
case = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True, no_output=True, log=False)

# Add a new PQ load
bus_idx = np.where(case.Bus.idx == 4)[0]
if bus_idx.size == 0:
    raise ValueError("Bus 4 not found.")
bus_idx = bus_idx[0]

case.PQ.add('PQ_VERIFY_040_D', bus=bus_idx, p0=0.014, q0=0.008)

# Set slack-bus voltage target and rerun power flow
slack_idx = case.Slack.idx[0]
case.Slack.set_vtarget(1.015)  # Set the voltage target
andes.run_power_flow(case)

# Report slack-bus voltage
slack_voltage = case.Bus.v[slack_idx]
slack_setpoint = 1.015

# Find the two lowest voltage buses
voltage_data = case.Bus.v
bus_ids = case.Bus.idx
lowest_voltage_indices = np.argsort(voltage_data)[:2]

# Prepare selected bus IDs and voltages
selected_bus_ids = bus_ids[lowest_voltage_indices].tolist()
selected_voltages = [round(voltage_data[idx], 6) for idx in lowest_voltage_indices]

# Plot bus voltages
plt.figure()
plt.plot(case.Bus.idx, voltage_data, marker='o')
plt.title('Bus Voltages')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.savefig('scenario_040_turn3_line.png')
plt.close()

# Prepare the result JSON
RESULT_JSON = {
    'slack_setpoint': slack_setpoint,
    'slack_voltage': round(slack_voltage, 6),
    'selected_bus_ids': selected_bus_ids,
    'selected_voltages': selected_voltages,
    'plot_file': 'scenario_040_turn3_line.png'
}

# Print the result JSON
print("RESULT_JSON=", RESULT_JSON)