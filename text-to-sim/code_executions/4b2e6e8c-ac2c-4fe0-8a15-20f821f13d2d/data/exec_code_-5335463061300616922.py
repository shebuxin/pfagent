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

# Max voltage (p.u.)
max_v = np.nanmax(v)
max_idx = int(np.nanargmax(v))
bus_id = idx[max_idx]

print(f"Bus {bus_id} Has the Highest Voltage: {max_v:.4f} p.u.")