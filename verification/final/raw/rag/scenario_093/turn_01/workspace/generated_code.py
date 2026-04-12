
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

# required_dependencies: andes,numpy
import json
import andes
import numpy as np
import os

script_dir = os.getcwd()
case = os.path.join(script_dir, "verify_pjm5_093.json")
ssa = andes.load(case, setup=False, no_output=True, log=False)

ssa.setup()

ssa.PFlow.run()

bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)
bus_v = np.asarray(ssa.Bus.v.v, dtype=float)
line_ids = [str(item) for item in np.asarray(ssa.Line.idx.v)]
line_metric = np.abs(np.asarray(ssa.Line.a1.e, dtype=float))

result_json = {}
top_k = 2
selected_idx = np.argsort(line_metric)[-top_k:][::-1]
result_json["selected_line_ids"] = [str(line_ids[i]) for i in selected_idx]
result_json["selected_line_metrics"] = [round(float(line_metric[i]), 6) for i in selected_idx]

print("RESULT_JSON=" + json.dumps(result_json, sort_keys=True))