# required_dependencies: andes,numpy
import os
import numpy as np
import andes

case_path = "ieee39.xlsx"
if not os.path.exists(case_path):
    raise FileNotFoundError(f"Case file not found: {case_path}")

ssa = andes.load(
    case_path,
    setup=False,
    no_output=True,
    default_config=False
)

ssa.setup()
ssa.PFlow.run()

v_mag = ssa.Bus.v.v
max_idx = int(np.nanargmax(v_mag))
bus_id = ssa.Bus.idx.v[max_idx]
max_v = v_mag[max_idx]

print(f"Bus {bus_id} has the Highest Voltage: {max_v:.4f} p.u.")
