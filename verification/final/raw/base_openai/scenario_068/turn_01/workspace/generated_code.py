
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

# required_dependencies: pandas
import andes
import pandas as pd

# Load the Kundur full case
case = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Run the power flow
andes.run("powerflow", ssa)

# Get the bus voltages
bus_voltages = pd.DataFrame({
    'bus_id': ssa.Bus.idx.v,
    'voltage': ssa.Bus.vmag.v
})

# Find the two lowest voltage buses
lowest_buses = bus_voltages.nsmallest(2, 'voltage').sort_values('voltage')

# Prepare the result
RESULT_JSON = {
    "selected_bus_ids": lowest_buses['bus_id'].tolist(),
    "selected_voltages": [round(v, 6) for v in lowest_buses['voltage'].tolist()]
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")