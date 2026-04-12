
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

# required_dependencies: andes,numpy,matplotlib
import json
import andes
import numpy as np
import os
import matplotlib.pyplot as plt

ssa = andes.load(
    andes.get_case("ieee39/ieee39.xlsx"),
    setup=False,
    no_output=True,
    log=False,
)

ssa.setup()

scale_factor = 1.02
ssa.PQ.set(src="p0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.p0.v)
ssa.PQ.set(src="q0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.q0.v)

pv_setpoint = 1.005
ssa.PV.set(src="v0", idx=[ssa.PV.idx.v[0]], attr="v", value=[pv_setpoint])

ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
pv_bus = int(ssa.PV.bus.v[0])
pv_pos = int(np.where(bus_ids == pv_bus)[0][0])
pv_voltage = round(float(bus_v[pv_pos]), 6)

plot_file = "scenario_037_turn3_bar.png"
plt.figure(figsize=(10, 4))
plt.bar(bus_ids, bus_v, width=0.8)
plt.xlabel("Bus ID")
plt.ylabel("Voltage Magnitude (p.u.)")
plt.title("Bus Voltage Profile")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plot_file, bbox_inches='tight')
plt.close()

result_json = {}
result_json["scale_factor"] = round(float(scale_factor), 6)
result_json["max_bus"] = int(bus_ids[int(np.argmax(bus_v))])
result_json["max_voltage"] = round(float(bus_v[int(np.argmax(bus_v))]), 6)
result_json["min_bus"] = int(bus_ids[int(np.argmin(bus_v))])
result_json["min_voltage"] = round(float(bus_v[int(np.argmin(bus_v))]), 6)
result_json["plot_file"] = plot_file

print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))