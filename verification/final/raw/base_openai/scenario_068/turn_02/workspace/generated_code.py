
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

# Add a new PQ load at bus 4
bus_idx = np.where(ssa.Bus.idx.v == 4)[0][0]
load_idx = ssa.PQ.add(idx='PQ_VERIFY_068_D', bus=bus_idx, p0=0.014, q0=0.008)

# Rerun the power flow
andes.run("powerflow", ssa)

# Get the slack bus information
slack_bus_idx = ssa.Slack.idx.v[0]
slack_voltage = ssa.Bus.vmag.v[slack_bus_idx]

# Find all buses below 0.940 p.u.
low_voltage_buses = pd.DataFrame({
    'bus_id': ssa.Bus.idx.v,
    'voltage': ssa.Bus.vmag.v
})

buses_below_threshold = low_voltage_buses[low_voltage_buses['voltage'] < 0.940]

# Prepare the result
RESULT_JSON = {
    "added_load_idx": load_idx,
    "slack_bus": slack_bus_idx,
    "slack_voltage": round(slack_voltage, 6),
    "threshold": 0.940,
    "selected_bus_ids": buses_below_threshold['bus_id'].tolist(),
    "selected_count": int(buses_below_threshold.shape[0])
}

# Print the result
print(f"RESULT_JSON={RESULT_JSON}")