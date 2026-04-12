```python
# required_dependencies: andes,json
import andes
import json
import os

def _round_float(value):
    return round(float(value), 6)

cwd = os.getcwd()
case = os.path.join(cwd, "verify_pjm5_096.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

scale_factor = 1.06
ssa.PQ.set(src="p0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.p0.v)
ssa.PQ.set(src="q0", idx=ssa.PQ.idx.v, attr="v", value=scale_factor * ssa.PQ.q0.v)
ssa.PFlow.run()

line_ids = ssa.Line.idx.v
abs_a1 = abs(ssa.Line.a1.e)
angle_threshold = 0.15
mask = abs_a1 > angle_threshold
result = {
    "scale_factor": _round_float(1.06),
    "angle_threshold": _round_float(angle_threshold),
    "selected_line_ids": [str(value) for value in line_ids[mask]],
    "selected_count": int(mask.sum()),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```