
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

# Load the built-in IEEE 14 full case
case = andes.get_case("ieee14/ieee14_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 9 (previously added)
load_idx = 'PQ_VERIFY_014_D'
p0 = 0.018
q0 = 0.012
ssa.PQ.add(load_idx, bus=9, p0=p0, q0=q0)

# Set the slack-bus voltage target to 1.035
slack_bus_idx = ssa.Slack.idx.v[0]
ssa.Slack.vref.set(slack_bus_idx, 1.035)

# Run power flow analysis after setting the slack voltage target
andes.run_power_flow(ssa)

# Identify the slack bus voltage
slack_voltage = round(float(ssa.Bus.vmag.v[ssa.Bus.idx.v == slack_bus_idx]), 6)
slack_setpoint = 1.035

# Extract bus voltages and their corresponding IDs
bus_ids = ssa.Bus.idx.v
voltages = ssa.Bus.vmag.v

# Combine bus ids and their voltages into a list of tuples and sort
bus_voltage_pairs = list(zip(bus_ids, voltages))
sorted_buses = sorted(bus_voltage_pairs, key=lambda x: x[1])

# Select the 4 lowest-voltage buses
lowest_buses = sorted_buses[:4]
selected_bus_ids = [int(bus[0]) for bus in lowest_buses]
selected_voltages = [round(float(bus[1]), 6) for bus in lowest_buses]

# Save a line plot of bus voltages
plt.figure()
plt.plot(bus_ids, voltages, marker='o')
plt.title('Bus Voltages')
plt.xlabel('Bus IDs')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.savefig('scenario_014_turn3_line.png')

# Print the result in the required format
RESULT_JSON = {
    "slack_setpoint": slack_setpoint,
    "slack_voltage": slack_voltage,
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages,
    "plot_file": 'scenario_014_turn3_line.png'
}
print(f"RESULT_JSON={RESULT_JSON}")