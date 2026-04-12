
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

# Load the built-in IEEE 39 case
case_path = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case_path, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 15
bus_index = 15
load_idx = 'PQ_VERIFY_042_D'
p0 = 0.018
q0 = 0.012
ssa.PQ.add(bus=bus_index, idx=load_idx, p0=p0, q0=q0)

# Set the slack-bus voltage target to 1.030
slack_bus_idx = ssa.Bus.idx.v[ssa.Bus.slack.v]
ssa.Bus.Vmag.v[slack_bus_idx] = 1.030

# Run power flow
andes.run(ssa)

# Extract slack bus information
slack_voltage = ssa.Bus.Vmag.v[ssa.Bus.slack.v]
slack_setpoint = 1.030

# Extract bus voltages and sort for lowest voltage buses
bus_ids = ssa.Bus.idx.v
voltages = ssa.Bus.Vmag.v
voltages_buses = sorted(zip(bus_ids, voltages), key=lambda x: x[1])[:4]
selected_bus_ids = [int(bus_id) for bus_id, _ in voltages_buses]
selected_voltages = [round(voltage, 6) for _, voltage in voltages_buses]

# Plot bus voltages
plt.figure()
plt.plot(bus_ids, voltages, marker='o', linestyle='-', color='b')
plt.title('Bus Voltages')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.savefig('scenario_042_turn3_line.png')
plt.close()

# Print the result in the specified format
RESULT_JSON = {
    "slack_setpoint": slack_setpoint,
    "slack_voltage": round(slack_voltage, 6),
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages,
    "plot_file": 'scenario_042_turn3_line.png'
}
print(f"RESULT_JSON={RESULT_JSON}")