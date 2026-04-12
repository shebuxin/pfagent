```python
# required_dependencies: andes
import andes

# Load the IEEE 39 case
case = andes.get_case("ieee39/ieee39.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Add a new PQ load at bus 20
added_load_idx = "PQ_VERIFY_032_A"
ssa.PQ.add(bus=20, p0=0.019, q0=0.012, id=added_load_idx)

# Rerun power flow
andes.run_power_flow(ssa)

# Extract buses below 0.980 p.u. and the minimum voltage bus
threshold = 0.980
voltages = ssa.bus.Vmag.v
bus_ids = ssa.bus.idx.v

selected_bus_ids = [int(bus_ids[i]) for i in range(len(voltages)) if voltages[i] < threshold]
selected_count = len(selected_bus_ids)

# Finding the minimum voltage bus
min_bus_index = voltages.index(min(voltages))
min_bus = int(bus_ids[min_bus_index])
min_voltage = round(float(voltages[min_bus_index]), 6)

# Prepare Result JSON
RESULT_JSON = {
    "added_load_idx": added_load_idx,
    "added_load_bus": 20,
    "threshold": round(threshold, 6),
    "selected_bus_ids": sorted(selected_bus_ids),
    "selected_count": selected_count,
    "min_bus": min_bus,
    "min_voltage": min_voltage
}

print("RESULT_JSON=", RESULT_JSON)
```