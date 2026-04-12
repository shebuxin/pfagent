
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

# required_dependencies: andes
import andes
import os
import json

# Load the case from the uploaded file
script_dir = os.getcwd()
case_filename = "verify_ieee14_017.xlsx"
case_path = os.path.join(script_dir, case_filename)

# Load the case setup
ssa = andes.load(case_path, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 9
new_load_idx = 'PQ_VERIFY_017_A'
p0 = 0.016
q0 = 0.01
bus_idx = np.where(ssa.Bus.idx.v == 9)[0][0]  # Find the index of bus 9
ssa.PQ.add(idx=new_load_idx, bus=bus_idx, p0=p0, q0=q0)

# Run power flow analysis with the added load
andes.run_power_flow(ssa)

# Define the voltage threshold
threshold = 1.010

# Get all buses below the threshold
voltages = ssa.Bus.V.v
selected_bus_indices = np.where(voltages < threshold)[0]  # Get indices of buses below threshold
selected_bus_ids = ssa.Bus.idx.v[selected_bus_indices].tolist()

# Find the minimum voltage bus
min_voltage = voltages.min()
min_bus_idx = ssa.Bus.idx.v[np.where(voltages == min_voltage)[0][0]]

# Prepare the result JSON
result_json = {
    "added_load_idx": new_load_idx,
    "added_load_bus": bus_idx,
    "threshold": threshold,
    "selected_bus_ids": sorted(selected_bus_ids),
    "selected_count": len(selected_bus_ids),
    "min_bus": int(min_bus_idx),
    "min_voltage": round(min_voltage, 6),
}

# Print the result JSON
print(f"RESULT_JSON={json.dumps(result_json)}")