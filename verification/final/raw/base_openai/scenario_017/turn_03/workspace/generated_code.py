
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
case_filename = "verify_ieee14_017.xlsx"
case_path = os.path.join(script_dir, case_filename)

# Load the case setup
ssa = andes.load(case_path, setup=True, no_output=True, log=False)

# Add the PQ load if not already added in previous steps
new_load_idx = 'PQ_VERIFY_017_A'
p0 = 0.016
q0 = 0.01
bus_idx = 9  # Bus 9
ssa.PQ.add(idx=new_load_idx, bus=bus_idx, p0=p0, q0=q0)

# Scale every PQ load by a factor of 1.050
scale_factor = 1.050
ssa.PQ.p0.v *= scale_factor
ssa.PQ.q0.v *= scale_factor

# Run power flow analysis with scaled loads
andes.run_power_flow(ssa)

# Get the bus voltages
voltages = ssa.Bus.V.v

# Identify max and min voltages
max_voltage = voltages.max()
max_bus = ssa.Bus.idx.v[np.where(voltages == max_voltage)[0][0]]
min_voltage = voltages.min()
min_bus = ssa.Bus.idx.v[np.where(voltages == min_voltage)[0][0]]

# Create a line plot of bus voltage magnitude
plt.figure()
plt.plot(ssa.Bus.idx.v, voltages, marker='o')
plt.title('Bus Voltage Magnitudes')
plt.xlabel('Bus ID')
plt.ylabel('Voltage Magnitude (p.u.)')
plt.grid()
plot_file = 'scenario_017_turn3_line.png'
plt.savefig(plot_file)
plt.close()

# Prepare the result JSON
result_json = {
    "scale_factor": round(scale_factor, 6),
    "max_bus": int(max_bus),
    "max_voltage": round(max_voltage, 6),
    "min_bus": int(min_bus),
    "min_voltage": round(min_voltage, 6),
    "plot_file": plot_file,
}

# Print the result JSON
print(f"RESULT_JSON={json.dumps(result_json)}")