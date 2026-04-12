import andes
import numpy as np

ssa = andes.load(andes.get_case('5bus/pjm5bus.xlsx'))

ssa.PFlow.run()


bus_voltages = ssa.Bus.v.v
bus_ids = ssa.Bus.idx.v

index_max = int(np.nanargmax(bus_voltages))
index_min = int(np.nanargmin(bus_voltages))

print(f"Max Voltage: Bus {bus_ids[index_max]} = {bus_voltages[index_max]:.4f} p.u.")
print(f"Min Voltage: Bus {bus_ids[index_min]} = {bus_voltages[index_min]:.4f} p.u.")


# Print maximum and minimum bus voltage