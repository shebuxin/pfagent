
# Auto-capture matplotlib plt.show() for headless execution and store plots in output/.
try:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _plot_output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(_plot_output_dir, exist_ok=True)
    _existing_plots = [
        name for name in os.listdir(_plot_output_dir)
        if name.startswith("plot_") and name.endswith(".png")
    ]
    _plot_counter = len(_existing_plots)

    def _streamlit_safe_show(*args, **kwargs):
        global _plot_counter
        saved_paths = []
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            _plot_counter += 1
            plot_path = os.path.join(_plot_output_dir, f"plot_{_plot_counter}.png")
            fig.savefig(plot_path, bbox_inches="tight")
            saved_paths.append(plot_path)
        if saved_paths:
            print("Saved plot(s):")
            for path in saved_paths:
                print(f"- {path}")
        plt.close("all")

    plt.show = _streamlit_safe_show
except Exception:
    pass

# required_dependencies: andes,matplotlib,numpy,pandas

import os

# Get the directory where this script is located
script_dir = os.path.realpath(os.path.join(os.path.split(__file__)[0]))

# Build the path to the case file
case_path = os.path.join(script_dir, 'ieee39.xlsx')

# Load the IEEE 39-bus case
import pandas as pd
import andes

net = andes.load(
    case_path,
    setup=True,     # immediately setup the network
    check=True,      # check the network for common errors
    log_level="ERROR"  # suppress unnecessary output
)

# Run the power flow
net.PFlow.run()

# Find buses with max and min voltage
v = net.bus.v.v  # voltage magnitudes
idx = net.bus.index  # bus indices
max_idx = int(idx[v.argmax()])
min_idx = int(idx[v.argmin()])
max_v = float(v.max())
min_v = float(v.min())

print(f"Bus {max_idx} has the highest voltage: {max_v:.4f} p.u.")
print(f"Bus {min_idx} has the lowest voltage: {min_v:.4f} p.u.")
print(f"Voltage difference: {max_v - min_v:.4f} p.u.")


# Plot all bus voltages
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.bar(idx, v, color='blue')
plt.xlabel('Bus Index')
plt.ylabel('Voltage (p.u.)')
plt.title('Bus Voltages in the IEEE 39-Bus System')
plt.xticks(idx, rotation=90)
plt.tight_layout()
plt.show()