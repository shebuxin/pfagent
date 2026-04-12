```python
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

scale_factor = 1.05
ssa.PQ.set(src="p0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.p0.v)
ssa.PQ.set(src="q0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.q0.v)

ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
line_ids = [str(item) for item in np.asarray(ssa.Line.idx.v)]
line_metric = np.abs(np.asarray(ssa.Line.a1.e, dtype=float))

result_json = {}
angle_threshold = 0.12
selected_mask = line_metric > angle_threshold
result_json["scale_factor"] = round(float(scale_factor), 6)
result_json["angle_threshold"] = round(float(angle_threshold), 6)
result_json["selected_line_ids"] = [str(item) for item, keep in zip(line_ids, selected_mask) if keep]
result_json["selected_count"] = int(np.sum(selected_mask))

print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))
```