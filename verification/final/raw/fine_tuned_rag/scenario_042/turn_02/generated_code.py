# required_dependencies: andes,numpy
import json
import andes
import numpy as np

ssa = andes.load(
    andes.get_case("ieee39/ieee39.xlsx"),
    setup=False,
    no_output=True,
    log=False,
)

ssa.add(
    "PQ",
    param_dict={
        "bus": 15,
        "idx": "PQ_VERIFY_042_D",
        "p0": 0.018,
        "q0": 0.012,
    },
)

ssa.setup()

ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
slack_bus = int(ssa.Slack.bus.v[0])
slack_pos = int(np.where(bus_ids == slack_bus)[0][0])
slack_voltage = round(float(bus_v[slack_pos]), 6)

result_json = {}
threshold = 0.97
below_mask = bus_v < threshold
result_json["added_load_idx"] = "PQ_VERIFY_042_D"
result_json["slack_bus"] = slack_bus
result_json["slack_voltage"] = slack_voltage
result_json["threshold"] = round(float(threshold), 6)
result_json["selected_bus_ids"] = [int(item) for item in bus_ids[below_mask]]
result_json["selected_count"] = int(np.sum(below_mask))

print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))