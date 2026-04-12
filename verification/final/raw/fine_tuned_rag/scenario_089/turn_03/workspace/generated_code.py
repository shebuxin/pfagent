
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

# required_dependencies: andes,numpy
import json
import andes
import numpy as np

ssa = andes.load(
    andes.get_case("5bus/pjm5bus.json"),
    setup=False,
    no_output=True,
    log=False,
)

ssa.setup()

scale_factor = 1.03
ssa.PQ.set(src="p0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.p0.v)
ssa.PQ.set(src="q0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.q0.v)

slack_setpoint = 1.0
ssa.Slack.set(src="v0", idx=[ssa.Slack.idx.v[0]], attr="v", value=[slack_setpoint])

ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
slack_bus = int(ssa.Slack.bus.v[0])
slack_pos = int(np.where(bus_ids == slack_bus)[0][0])
slack_voltage = round(float(bus_v[slack_pos]), 6)
line_ids = [str(item) for item in np.asarray(ssa.Line.idx.v)]
line_metric = np.abs(np.asarray(ssa.Line.a1.e, dtype=float))

result_json = {}
angle_threshold = 0.08
selected_mask = line_metric > angle_threshold
result_json["slack_setpoint"] = round(float(slack_setpoint), 6)
result_json["scale_factor"] = round(float(scale_factor), 6)
result_json["angle_threshold"] = round(float(angle_threshold), 6)
result_json["selected_line_ids"] = [str(item) for item, keep in zip(line_ids, selected_mask) if keep]
result_json["selected_count"] = int(np.sum(selected_mask))

print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))