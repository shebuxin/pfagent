# required_dependencies: andes,json
import andes
import json
import os
import numpy as np

def _round_float(value):
    return round(float(value), 6)

cwd = os.getcwd()
case = os.path.join(cwd, "verify_pjm5_093.json")
ssa = andes.load(case, setup=True, no_output=True, log=False)

ssa.PFlow.run()

line_ids = np.asarray(ssa.Line.idx.v)
abs_a1 = np.abs(np.asarray(ssa.Line.a1.e, dtype=float))
top_k = 2
rank_indices = np.argsort(abs_a1)[-top_k:][::-1]
result = {
    "selected_line_ids": [str(line_ids[i]) for i in rank_indices],
    "selected_line_metrics": [_round_float(abs_a1[i]) for i in rank_indices],
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))