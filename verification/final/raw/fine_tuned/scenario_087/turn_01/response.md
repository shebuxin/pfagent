```python
# required_dependencies: numpy
import json
import numpy as np
import andes

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    andes.get_case("5bus/pjm5bus.json"),
    setup=False,
    no_output=True,
)

ssa.setup()
ssa.PFlow.run()

line_abs_a1 = np.abs(np.asarray(ssa.Line.a1.e, dtype=float))
top_k = 4
rank_indices = np.argsort(line_abs_a1)[-top_k:][::-1]
result = {
    "selected_line_ids": [str(ssa.Line.idx.v[i]) for i in rank_indices],
    "selected_line_metrics": [_round_float(line_abs_a1[i]) for i in rank_indices],
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```