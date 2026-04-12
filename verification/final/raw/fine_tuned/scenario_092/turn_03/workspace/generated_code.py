
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

# required_dependencies: numpy
import json
import numpy as np
import andes

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    andes.get_case("5bus/pjm5bus.json"),
    setup=False,
    no_output=True,
)

ssa.setup()
slack_setpoint = 1.03
ssa.Slack.set(src="v0", idx=[ssa.Slack.idx.v[0]], attr="v", value=[slack_setpoint])
scale_factor = 1.06
ssa.PQ.set(src="p0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.p0.v)
ssa.PQ.set(src="q0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.q0.v)
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
line_ids = np.asarray(ssa.Line.idx.v)
line_bus1 = np.asarray(ssa.Line.bus1.v)
line_bus2 = np.asarray(ssa.Line.bus2.v)
line_a1 = np.asarray(ssa.Line.a1.v, dtype=float)
slack_bus = int(ssa.Slack.bus.v[0])
slack_index = int(np.where(bus_ids == slack_bus)[0][0])
slack_voltage = _round_float(bus_v[slack_index])
angle_threshold = 0.15
mask = np.isin(line_bus1, bus_ids)
candidate_ids = line_ids[mask]
candidate_a1 = line_a1[mask]
candidate_bus1 = line_bus1[mask]
candidate_bus2 = line_bus2[mask]
bus_index_map = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
candidate_a1 = np.asarray([abs(candidate_a1[i] * (bus_v[bus_index_map[candidate_bus1[i]]] - bus_v[bus_index_map[candidate_bus2[i]]])) for i in range(len(candidate_ids))])
mask = candidate_a1 > angle_threshold
result = {
    "slack_setpoint": _round_float(1.03),
    "scale_factor": _round_float(1.06),
    "angle_threshold": _round_float(angle_threshold),
    "selected_line_ids": [int(value) for value in candidate_ids[mask]],
    "selected_count": int(np.sum(mask)),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))