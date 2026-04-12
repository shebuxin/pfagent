
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

# Load the Kundur full case
case_path = andes.get_case("kundur/kundur_full.json")
ssa = andes.load(case_path, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 7 if not already added
pq_load_idx = 'PQ_VERIFY_070_D'
bus_7_idx = 7  # Bus 7 index
p0 = 0.018
q0 = 0.012

# Add the PQ load
ssa.PQ.add(buses=bus_7_idx, p0=p0, q0=q0, idx=pq_load_idx)

# Set slack-bus voltage target
voltage_target = 1.010
ssa.Bus.v_target.v[ssa.Bus.slack_flag.v] = voltage_target

# Setup modified case
ssa.setup()

# Run power flow
andes.run_power_flow(ssa)

# Get slack bus voltage
slack_bus_idx = ssa.bus.v[ssa.Bus.slack_flag.v].idx.v[0]
slack_voltage = round(ssa.Bus.v[slack_bus_idx], 6)

# Find the lowest-voltage buses
voltages = ssa.Bus.v
lowest_voltage_indices = voltages.argsort()[:4]
selected_bus_ids = ssa.Bus.idx.v[lowest_voltage_indices].tolist()
selected_voltages = [round(voltages[idx], 6) for idx in lowest_voltage_indices]

# Save line plot of bus voltages
plt.figure()
plt.plot(ssa.Bus.idx.v, voltages, marker='o')
plt.title('Bus Voltages')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.savefig('scenario_070_turn3_line.png')
plt.close()

# Prepare the result JSON
RESULT_JSON = {
    "slack_setpoint": voltage_target,
    "slack_voltage": slack_voltage,
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages,
    "plot_file": 'scenario_070_turn3_line.png'
}

# Print the result
print("RESULT_JSON=", RESULT_JSON)