
try:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _verification_output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(_verification_output_dir, exist_ok=True)
    _verification_existing = [
        name for name in os.listdir(_verification_output_dir)
        if name.startswith("plot_") and name.endswith(".png")
    ]
    _verification_plot_counter = len(_verification_existing)

    def _verification_safe_show(*args, **kwargs):
        global _verification_plot_counter
        saved_paths = []
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            _verification_plot_counter += 1
            plot_path = os.path.join(_verification_output_dir, f"plot_{_verification_plot_counter}.png")
            fig.savefig(plot_path, bbox_inches="tight")
            saved_paths.append(plot_path)
        if saved_paths:
            print("Saved plot(s):")
            for path in saved_paths:
                print(f"- {path}")
        plt.close("all")

    plt.show = _verification_safe_show
except Exception:
    pass

# required_dependencies: numpy,json
import os
import numpy as np
import json
import andes

# Load the case
script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_ieee14_023.xlsx")
ssa = andes.load(case, setup=True, no_output=True, log=False)

# Set the first PV voltage target
first_pv_idx = ssa.PV.idx.v[0]  # Get index of the first PV
ssa.PV.v_target.set_value(first_pv_idx, 1.010)

# Rerun power flow
andes.power_flow(ssa)

# Get the affected PV bus voltage
pv_bus = int(ssa.PV.bus.v[first_pv_idx])
pv_voltage = float(ssa.PV.v.v[first_pv_idx])
pv_setpoint = 1.010  # Setpoint we just assigned

# Count how many buses are above 1.015 p.u.
bus_voltages = ssa.Bus.v.v
selected_count = int(np.sum(bus_voltages > 1.015))

# Prepare the RESULT_JSON output
result_json = {
    "pv_bus": pv_bus,
    "pv_setpoint": round(pv_setpoint, 6),
    "pv_voltage": round(pv_voltage, 6),
    "selected_count": selected_count
}

# Print the result
print(f'RESULT_JSON={json.dumps(result_json)}')