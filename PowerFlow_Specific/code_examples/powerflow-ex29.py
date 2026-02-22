import andes
import numpy as np


# Voltage limits
V_MIN = 0.95
V_MAX = 1.05


ssa = andes.load(
    andes.get_case('kundur/kundur_full.xlsx')
)

# Add new load
ssa.add(model="PQ", param_dict=dict(bus=2, 
                                    idx="PQ_484", 
                                    p0=60/ssa.config.mva, 
                                    q0=100/ssa.config.mva
                                    ))

ssa.setup()
ssa.PFlow.run()

bus_voltages = ssa.Bus.v.v
bus_ids = ssa.Bus.idx.v

# Max and min voltages
index_max = int(np.nanargmax(bus_voltages))
index_min = int(np.nanargmin(bus_voltages))

if bus_voltages[index_max] > V_MAX or bus_voltages[index_min] < V_MIN:
    print("Voltage Violations Found")
    
    # Check for violations
    print(f"Voltage Violations (Outside {V_MIN:.2f} – {V_MAX:.2f} p.u.):")
    for bus_id, voltage in zip(bus_ids, bus_voltages):
        if voltage < V_MIN or voltage > V_MAX:
            print(f"Bus {bus_id}: {voltage:.4f} p.u.")
else:
    print("No Voltage Violations Found")


# Add new load to specific bus or new generation, or general operations of the base case (if a load goes down, are there any voltage violations)