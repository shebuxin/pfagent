# required_dependencies: andes,numpy
import json
import andes
import numpy as np
import os

script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_kundur_078.xlsx")
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

result_json = {}
threshold = 0.97
result_json["slack_bus"] = slack_bus
result_json["slack_setpoint"] = round(float(slack_setpoint), 6)
result_json["slack_voltage"] = slack_voltage
result_json["selected_count"] = int(np.sum(bus_v < threshold))

print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))