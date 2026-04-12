# required_dependencies: andes,json
import andes
import json
import os
import numpy as np

def _round_float(value):
    return round(float(value), 6)

cwd = os.getcwd()
case = os.path.join(cwd, "verify_pjm5_095.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

scale_factor = 1.05
ssa.PQ.set(src="p0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.p0.v)
ssa.PQ.set(src="q0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.q0.v)
ssa.PFlow.run()

line_ids = np.asarray(ssa.Line.idx.v)
abs_a1 = np.abs(np.asarray(ssa.Line.a1.e, dtype=float))
angle_threshold = 0.12
mask = abs_a1 > angle_threshold
result = {
    "scale_factor": _round_float(1.05),
    "angle_threshold": _round_float(angle_threshold),
    "selected_line_ids": [str(value) for value in line_ids[mask]],
    "selected_count": int(np.sum(mask)),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))