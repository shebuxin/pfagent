
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
import numpy as np

# Load the Kundur full case
case = andes.get_case("kundur/kundur_full.xlsx")
ssa = andes.load(case, setup=True)

# Run power flow
andes.solve()

# Extract the bus data
voltages = ssa.Bus.v.v
bus_ids = ssa.Bus.idx.v

# Get the slack bus voltage
slack_bus_idx = np.where(ssa.Bus.type.v == 2)[0][0]  # Slack bus is type 2
slack_voltage = round(voltages[slack_bus_idx], 6)

# Get top 4 highest-voltage buses
top_indices = np.argsort(voltages)[-4:][::-1]
selected_bus_ids = bus_ids[top_indices].tolist()
selected_voltages = [round(voltages[idx], 6) for idx in top_indices]

# Prepare the result JSON
RESULT_JSON = {
    "slack_bus": int(ssa.Bus.idx.v[slack_bus_idx]),
    "slack_voltage": slack_voltage,
    "selected_bus_ids": selected_bus_ids,
    "selected_voltages": selected_voltages
}

print(f"RESULT_JSON={RESULT_JSON}")