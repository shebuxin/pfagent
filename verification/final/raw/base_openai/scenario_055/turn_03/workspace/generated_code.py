
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
import json
import matplotlib.pyplot as plt

# Load the case from the uploaded file
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee39_055.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 10 (if not already added from previous step)
new_load_idx = 'PQ_VERIFY_055_D'
p0 = 0.016
q0 = 0.01
ssa.PQ.add(bus=10, idx=new_load_idx, p0=p0, q0=q0)

# Set slack bus voltage target
slack_setpoint = 1.020
slack_bus_idx = ssa.Bus.slack.v[0]
ssa.Bus.vmag.set(idx=slack_bus_idx, value=slack_setpoint)

# Prepare for power flow analysis
andes.setup(ssa)

# Run power flow analysis
andes.run_power_flow()

# Get Slack Bus voltage
slack_voltage = round(float(ssa.Bus.vmag.v[slack_bus_idx]), 6)

# Get bus voltages and sort them
bus_ids = ssa.Bus.idx.v
voltages = ssa.Bus.vmag.v
sorted_indices = sorted(range(len(voltages)), key=lambda i: voltages[i])
lowest_voltage_buses = sorted_indices[:3]

# Prepare selected bus information
selected_bus_ids = [int(bus_ids[i]) for i in lowest_voltage_buses]
selected_voltages = [round(float(voltages[i]), 6) for i in lowest_voltage_buses]

# Plot bus voltages
plt.figure(figsize=(10, 6))
plt.plot(bus_ids, voltages, marker='o')
plt.title('Bus Voltages')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.savefig('scenario_055_turn3_line.png')

# Prepare the result
RESULT_JSON = json.dumps({
    "slack_setpoint": slack_setpoint,
    "slack_voltage": slack_voltage,
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages,
    "plot_file": 'scenario_055_turn3_line.png'
})

# Print the result as JSON
print(f"RESULT_JSON={RESULT_JSON}")