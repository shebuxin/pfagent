# required_dependencies: andes,numpy
import json
import andes
import numpy as np

ssa = andes.load(
    andes.get_case("kundur/kundur_full.xlsx"),
    setup=False,
    no_output=True,
    log=False,
)

ssa.add(
    "PQ",
    param_dict={
        "bus": 6,
        "idx": "PQ_VERIFY_062_B",
        "p0": 0.014,
        "q0": 0.009,
    },
)

ssa.setup()

slack_setpoint = 1.0
ssa.Slack.set(src="v0", idx=[ssa.Slack.idx.v[0]], attr="v", value=[slack_setpoint])

ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
slack_bus = int(ssa.Slack.bus.v[0])
slack_pos = int(np.where(bus_ids == slack_bus)[0][0])
slack_voltage = round(float(bus_v[slack_pos]), 6)

result_json = {}
result_json["added_load_idx"] = "PQ_VERIFY_062_B"
result_json["max_bus"] = int(bus_ids[int(np.argmax(bus_v))])
result_json["max_voltage"] = round(float(bus_v[int(np.argmax(bus_v))]), 6)
result_json["min_bus"] = int(bus_ids[int(np.argmin(bus_v))])
result_json["min_voltage"] = round(float(bus_v[int(np.argmin(bus_v))]), 6)
result_json["total_pq_count"] = int(len(ssa.PQ.idx.v))

print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))