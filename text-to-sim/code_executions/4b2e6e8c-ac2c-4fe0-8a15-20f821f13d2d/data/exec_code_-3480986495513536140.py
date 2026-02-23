import os
import sys

import numpy as np

case_path = "ieee39.xlsx"

if not os.path.exists(case_path):
    print(f"Case file not found: {case_path}")
    sys.exit(1)

import andes

ssa = andes.load(case_path, setup=False, no_output=True, default_config=False)
ssa.setup()
ssa.PFlow.run()

v = ssa.Bus.v.v
idx = ssa.Bus.idx.v

# Min voltage (p.u.)
min_v = np.nanmin(v)
min_idx = int(np.nanargmin(v))
bus_id = idx[min_idx]

print(f"Bus {bus_id} Has the Lowest Voltage: {min_v:.4f} p.u.")