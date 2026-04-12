
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

script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_027.xlsx")
ssa = andes.load(case, setup=False, no_output=True, log=False)

ssa.add(
    "PQ",
    param_dict={
        "bus": 5,
        "idx": "PQ_VERIFY_027_D",
        "p0": 0.016,
        "q0": 0.01,
    },
)

ssa.setup()

slack_setpoint = 1.025
ssa.Slack.set(src="v0", idx=[ssa.Slack.idx.v[0]], attr="v", value=[slack_setpoint])

ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
slack_bus = int(ssa.Slack.bus.v[0])
slack_pos = int(np.where(bus_ids == slack_bus)[0][0])
slack_voltage = round(float(bus_v[slack_pos]), 6)

plot_file = "scenario_027_turn3_line.png"
plt.figure(figsize=(10, 4))
plt.plot(bus_ids, bus_v, marker="o")
plt.xlabel("Bus ID")
plt.ylabel("Voltage Magnitude (p.u.)")
plt.title("Bus Voltage Profile")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plot_file, bbox_inches='tight')
plt.close()

result_json = {}
top_k = 3
selected_idx = np.argsort(bus_v)[:top_k]
result_json["slack_setpoint"] = round(float(slack_setpoint), 6)
result_json["slack_voltage"] = slack_voltage
result_json["selected_bus_ids"] = [int(bus_ids[i]) for i in selected_idx]
result_json["selected_voltages"] = [round(float(bus_v[i]), 6) for i in selected_idx]
result_json["plot_file"] = plot_file

print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))