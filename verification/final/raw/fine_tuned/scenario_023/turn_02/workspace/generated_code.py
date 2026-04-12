
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
import numpy as np

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    "verify_ieee14_023.xlsx",
    setup=False,
    no_output=True,
)

ssa.setup()
pv_setpoint = 1.01
ssa.PV.set(src="v0", idx=[ssa.PV.idx.v[0]], attr="v", value=[pv_setpoint])
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
pv_bus = int(ssa.PV.bus.v[0])
pv_index = int(np.where(bus_ids == pv_bus)[0][0])
threshold = 1.015
result = {
    "pv_bus": pv_bus,
    "pv_setpoint": _round_float(1.01),
    "pv_voltage": _round_float(bus_v[pv_index]),
    "selected_count": int(np.sum(bus_v > threshold)),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))