# required_dependencies: andes,numpy
import json
import andes
import numpy as np
import os

script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_pjm5_099.json")
ssa = andes.load(case, setup=False, no_output=True, log=False)

ssa.setup()

slack_setpoint = 1.02
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
top_k = 4
selected_idx = np.argsort(line_metric)[-top_k:][::-1]
result_json["slack_setpoint"] = round(float(slack_setpoint), 6)
result_json["slack_voltage"] = slack_voltage
result_json["selected_line_ids"] = [str(line_ids[i]) for i in selected_idx]
result_json["selected_line_metrics"] = [round(float(line_metric[i]), 6) for i in selected_idx]

print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))