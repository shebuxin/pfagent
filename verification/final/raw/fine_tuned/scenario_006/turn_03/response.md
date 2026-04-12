import json
from typing import Any

import andes
import numpy as np

def _round_float(value: float) -> float:
    return round(float(value), 6)

ssa = andes.load(
    andes.get_case("ieee14/ieee14_full.xlsx"),
    setup=False,
    no_output=True,
)

ssa.add(
    "PQ",
    param_dict={
        "bus": 5,
        "idx": "PQ_VERIFY_006_B",
        "p0": 0.014,
        "q0": 0.009,
    },
)

ssa.setup()
slack_setpoint = 1.025
ssa.Slack.set(src="v0", idx=[ssa.Slack.idx.v[0]], attr="v", value=[slack_setpoint])
ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
max_index = int(np.argmax(bus_v))
min_index = int(np.argmin(bus_v))
result = {
    "max_bus": int(bus_ids[max_index]),
    "max_voltage": _round_float(bus_v[max_index]),
    "min_bus": int(bus_ids[min_index]),
    "min_voltage": _round_float(bus_v[min_index]),
}
result["added_load_idx"] = "PQ_VERIFY_006_B"
result["total_pq_count"] = int(len(ssa.PQ.idx.v))
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```