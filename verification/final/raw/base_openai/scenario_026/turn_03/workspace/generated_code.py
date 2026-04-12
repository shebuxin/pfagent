
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
import os
import numpy as np
import matplotlib.pyplot as plt

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_026.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add the PQ load if not already added (from the previous step)
bus_idx = np.where(ssa.Bus.idx.v == 4)[0]
if bus_idx.size == 0:
    raise ValueError("Bus 4 not found.")

# Set the slack-bus voltage target
slack_bus_idx = ssa.Bus.slack.v
ssa.Bus.Vtarget.set(slack_bus_idx, 1.020)

# Rerun power flow analysis
andes.setup(ssa)
andes.run_power_flow(ssa)

# Extract slack voltage
slack_voltage = round(float(ssa.Bus.V.v[slack_bus_idx - 1]), 6)  # -1 for zero indexing

# Extract bus voltage data and identify the two lowest voltage buses
bus_ids = ssa.Bus.idx.v
voltages = ssa.Bus.V.v
voltage_data = list(zip(bus_ids, voltages))
lowest_voltage_buses = sorted(voltage_data, key=lambda x: x[1])[:2]

# Prepare result data for JSON
selected_bus_ids = [int(bus[0]) for bus in lowest_voltage_buses]
selected_voltages = [round(float(bus[1]), 6) for bus in lowest_voltage_buses]

# Create the plot of bus voltages
plt.figure(figsize=(10, 6))
plt.plot(bus_ids, voltages, marker='o')
plt.title('Bus Voltages')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.axhline(y=1.020, color='r', linestyle='--', label='Slack Target Voltage (1.020 p.u.)')
plt.legend()
plt.savefig('scenario_026_turn3_line.png')
plt.close()

# Create RESULT_JSON
RESULT_JSON = {
    'slack_setpoint': 1.020,
    'slack_voltage': slack_voltage,
    'selected_bus_ids': selected_bus_ids,
    'selected_voltages': selected_voltages,
    'plot_file': 'scenario_026_turn3_line.png'
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)