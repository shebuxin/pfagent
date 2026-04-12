
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
case = andes.load(andes.get_case("kundur/kundur_full.xlsx"), setup=True, no_output=True, log=False)

# Add a new PQ load at bus 6
load_idx = case.PQ.idx.add()  # Create a new load index
case.PQ.idx.v[load_idx] = 'PQ_VERIFY_058_A'
case.PQ.bus.v[load_idx] = 6
case.PQ.p0.v[load_idx] = 0.013
case.PQ.q0.v[load_idx] = 0.008

# Run power flow after adding the load
andes.run_powerflow(case)

# Scale every PQ load by a factor of 1.040
scale_factor = 1.040
for idx in range(case.PQ.idx.count):
    case.PQ.p0.v[idx] *= scale_factor
    case.PQ.q0.v[idx] *= scale_factor

# Run power flow after scaling loads
andes.run_powerflow(case)

# Retrieve bus voltages for plotting
bus_voltages = case.Bus.v

# Find max and min bus voltages
max_voltage = max(bus_voltages)
max_bus = int(bus_voltages.argmax())
min_voltage = min(bus_voltages)
min_bus = int(bus_voltages.argmin())

# Save line plot of bus voltage magnitude
plt.figure(figsize=(10, 5))
plt.plot(bus_voltages, marker='o')
plt.title('Bus Voltage Magnitude')
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')
plt.grid()
plt.savefig('scenario_058_turn3_line.png')
plt.close()

# Build the result JSON object
RESULT_JSON = {
    "scale_factor": scale_factor,
    "max_bus": max_bus,
    "max_voltage": round(max_voltage, 6),
    "min_bus": min_bus,
    "min_voltage": round(min_voltage, 6),
    "plot_file": 'scenario_058_turn3_line.png',
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")