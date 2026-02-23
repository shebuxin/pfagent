
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

# required_dependencies: andes, numpy

import andes
import numpy as np


ssa = andes.load(
    andes.get_case("ieee14/ieee14.raw"),
    setup=True,
    no_output=True,
    log=False,
)
ssa.PFlow.run()

v = ssa.Bus.v.v
bus = ssa.Bus.idx.v

max_bus = int(np.argmax(v))
max_v = v[max_bus]

min_bus = int(np.argmin(v))
min_v = v[min_bus]

print(f"Bus {max_bus} has Highest Voltage: {max_v:.4f} p.u.")
print(f"Bus {min_bus} has Lowest Voltage: {min_v:.4f} p.u.")