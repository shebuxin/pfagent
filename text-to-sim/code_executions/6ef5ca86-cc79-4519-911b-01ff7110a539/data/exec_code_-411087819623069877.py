
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

# required_dependencies: anodes, matplotlib, scipy

import anodes
import matplotlib.pyplot as plt


# Load and modify the power system model
ssa = anodes.load(
    anodes.get_case('ieee14/ieee14.raw'),
    setup=True, check=True
)

# Add new load and generation
ssa.add(model='Load', param=dict(bus=2, nth=0, p0=0.5, q0=0.2))
ssa.add(model='PV', param=dict(bus=14, p0=0.5, q0=0.2, sn=20))

ssa.setup()
ssa.PFlow.run()


# Compute and display bus voltage magnitudes
v_buses = ssa.Bus.v.v
i_buses = ssa.Bus.i.v
bus_ids = ssa.Bus.idx.v

max_v_idx = int(bus_ids[v_buses.argmax()])
min_v_idx = int(bus_ids[v_buses.argmin()])

print(f"Bus With Highest Voltage: {max_v_idx} ({v_buses.max():.4f} p.u.)")
print(f"Bus With Lowest Voltage: {min_v_idx} ({v_buses.min():.4f} p.u.)")
print(f"Voltage Difference: {v_buses.max() - v_buses.min():.4f} p.u.")


# Plot bus voltages
plt.figure(figsize=(10, 6))
plt.bar(bus_ids, v_buses, color='skyblue')
plt.xlabel('Bus ID')
plt.ylabel('Voltage (p.u.)')
plt.title('Bus Voltages in the IEEE 14-Bus System')
plt.xticks(bus_ids, rotation=90)
plt.tight_layout()
plt.show()