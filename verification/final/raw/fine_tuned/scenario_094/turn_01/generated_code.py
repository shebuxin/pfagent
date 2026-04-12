# required_dependencies: andes,json
import andes
import json

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    "verify_pjm5_094.json",
    setup=False,
    no_output=True,
)

ssa.setup()
ssa.PFlow.run()

line_ids = ssa.Line.idx.v
abs_a1 = abs(ssa.Line.a1.e)
top_k = 3
rank_indices = abs_a1.argsort()[-top_k:][::-1]
result = {
    "selected_line_ids": [str(line_ids[i]) for i in rank_indices],
    "selected_line_metrics": [_round_float(abs_a1[i]) for i in rank_indices],
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))