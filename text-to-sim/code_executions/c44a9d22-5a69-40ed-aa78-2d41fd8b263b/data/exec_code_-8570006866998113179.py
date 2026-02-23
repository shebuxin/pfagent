
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

import andes
import numpy as np
import matplotlib.pyplot as plt
import os

# Uploaded case: use exact filename directly, not andes.get_case(...)
script_dir = os.getcwd()
case = os.path.join(script_dir, "ieee39.xlsx")
if not os.path.exists(case):
    raise FileNotFoundError(f"Missing uploaded case file: {case}")

ssa = andes.load(case, setup=True, no_output=True, log=False)
ssa.PFlow.run()

v = ssa.Bus.v.v
bus = ssa.Bus.idx.v
max_i = int(np.argmax(v))
min_i = int(np.argmin(v))

print(f"Bus {int(bus[max_i])} has highest voltage: {v[max_i]:.4f} p.u.")
print(f"Bus {int(bus[min_i])} has lowest voltage: {v[min_i]:.4f} p.u.")

os.makedirs("output", exist_ok=True)
plt.figure(figsize=(10, 4))
plt.plot(bus, v, marker="o")
plt.xlabel("Bus ID")
plt.ylabel("Voltage Magnitude (p.u.)")
plt.title("IEEE39 Voltage Profile")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/ieee39_voltage_profile.png", dpi=150)
plt.show()