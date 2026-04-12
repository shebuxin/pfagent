
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
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run power flow
andes.run()

# Retrieve bus voltages
bus_voltages = pd.DataFrame({
    'bus_id': ssa.bus.idx.v,
    'voltage': ssa.bus.vmag.v
})

# Find the 4 lowest-voltage buses
lowest_voltages = bus_voltages.nsmallest(4, 'voltage')

# Create the result JSON
result_json = {
    'selected_bus_ids': list(lowest_voltages['bus_id']),
    'selected_voltages': [round(v, 6) for v in list(lowest_voltages['voltage'])]
}

# Print the result
print(f'RESULT_JSON={result_json}')