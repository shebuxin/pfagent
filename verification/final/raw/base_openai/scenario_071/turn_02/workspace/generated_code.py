
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

# required_dependencies: andes,json
import andes
import json
import os
import numpy as np

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_071.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add new PQ load at bus 4
new_load_idx = 'PQ_VERIFY_071_A'
p0 = 0.01
q0 = 0.006
bus_index = np.where(ssa.Bus.id.v == 4)[0]

if bus_index.size == 0:
    raise ValueError("Bus 4 not found in the case.")
    
# Modify PQ load before running the setup
ssa.PQ.add_pq(bus=4, id=new_load_idx, p0=p0, q0=q0)

# Run power flow again
andes.run_power_flow(ssa)

# Retrieve bus voltages
voltage_data = ssa.Bus.voltage.v
bus_ids = ssa.Bus.id.v

# Threshold for reporting
threshold = 0.940

# Find buses below the threshold
selected_bus_ids = bus_ids[voltage_data < threshold]
selected_bus_ids = selected_bus_ids.tolist()

# Find the minimum voltage bus
min_voltage_idx = np.argmin(voltage_data)
min_bus = bus_ids[min_voltage_idx]
min_voltage = round(voltage_data[min_voltage_idx], 6)

# Prepare the result
result_json = {
    "added_load_idx": new_load_idx,
    "added_load_bus": 4,
    "threshold": round(threshold, 6),
    "selected_bus_ids": list(map(int, selected_bus_ids)),
    "selected_count": len(selected_bus_ids),
    "min_bus": int(min_bus),
    "min_voltage": float(min_voltage)
}

# Print the result in the required format
print(f"RESULT_JSON={json.dumps(result_json)}")