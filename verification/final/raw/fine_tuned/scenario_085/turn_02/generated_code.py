# required_dependencies: andes,json
import andes
import json
import numpy as np

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    andes.get_case("5bus/pjm5bus.json"),
    setup=True,
    no_output=True,
)

scale_factor = 1.03
ssa.PQ.set(src="p0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.p0.v)
ssa.PQ.set(src="q0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.q0.v)
ssa.PFlow.run()

line_ids = np.asarray(ssa.Line.idx.v)
abs_a1 = np.abs(np.asarray(ssa.Line.a1.e, dtype=float))
angle_threshold = 0.08
mask = abs_a1 > angle_threshold
result = {
    "scale_factor": _round_float(1.03),
    "angle_threshold": _round_float(angle_threshold),
    "selected_line_ids": [str(value) for value in line_ids[mask]],
    "selected_count": int(np.sum(mask)),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))