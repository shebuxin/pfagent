
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

# required_dependencies: andes,pandas
import andes
import pandas as pd
import os

# Load the uploaded case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_084.xlsx")
ssa = andes.load(case, setup=False, no_output=True, log=False)

# Add a new PQ load at bus 7
new_load_idx = 'PQ_VERIFY_084_D'
ssa.PQ.add(idx=new_load_idx, bus=7, p0=0.018, q0=0.012)

# Setup the model
andes.setup(ssa)

# Run power flow
andes.run()

# Retrieve slack bus information
slack_bus = ssa.bus.idx[ssa.bus.slack].tolist()[0]
slack_voltage = ssa.bus.vmag.v[ssa.bus.slack][0]

# Find every bus below 0.960 p.u.
threshold = 0.960
bus_voltages = pd.DataFrame({
    'bus_id': ssa.bus.idx.v,
    'voltage': ssa.bus.vmag.v
})

selected_buses = bus_voltages[bus_voltages['voltage'] < threshold]
selected_bus_ids = list(selected_buses['bus_id'])
selected_count = len(selected_bus_ids)

# Create the result JSON
result_json = {
    'added_load_idx': new_load_idx,
    'slack_bus': slack_bus,
    'slack_voltage': round(slack_voltage, 6),
    'threshold': threshold,
    'selected_bus_ids': selected_bus_ids,
    'selected_count': selected_count
}

# Print the result
print(f'RESULT_JSON={result_json}')