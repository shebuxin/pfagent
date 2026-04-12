# required_dependencies: andes
import andes

case = andes.load(andes.get_case("kundur/kundur_full.xlsx"), setup=True, no_output=True, log=False)
result = andes.run_power_flow(case)

max_bus_index = result.V.idx.v[np.argmax(result.V.mag.v)]
min_bus_index = result.V.idx.v[np.argmin(result.V.mag.v)]
max_voltage = round(np.max(result.V.mag.v), 6)
min_voltage = round(np.min(result.V.mag.v), 6)

RESULT_JSON = {
    "max_bus": int(max_bus_index),
    "max_voltage": float(max_voltage),
    "min_bus": int(min_bus_index),
    "min_voltage": float(min_voltage)
}

print("RESULT_JSON=", RESULT_JSON)