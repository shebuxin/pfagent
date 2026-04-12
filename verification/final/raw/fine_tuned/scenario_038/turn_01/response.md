```python
# required_dependencies: andes,json
import andes
import json

def _round_float(value):
    return round(float(value), 6)

ssa = andes.load(
    andes.get_case("ieee39/ieee39.xlsx"),
    setup=False,
    no_output=True,
)

ssa.setup()
ssa.PFlow.run()

bus_ids = ssa.Bus.idx.v
bus_v = ssa.Bus.v.v
max_index = int(int(np.argmax(bus_v)))
min_index = int(int(np.argmin(bus_v)))
result = {
    "max_bus": int(bus_ids[max_index]),
    "max_voltage": _round_float(bus_v[max_index]),
    "min_bus": int(bus_ids[min_index]),
    "min_voltage": _round_float(bus_v[min_index]),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
```