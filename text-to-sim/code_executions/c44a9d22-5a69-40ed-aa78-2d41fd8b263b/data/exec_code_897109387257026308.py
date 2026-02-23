
# Auto-capture matplotlib plt.show() for headless execution and store plots in output/.
try:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _plot_output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(_plot_output_dir, exist_ok=True)
    _existing_plots = [
        name for name in os.listdir(_plot_output_dir)
        if name.startswith("plot_") and name.endswith(".png")
    ]
    _plot_counter = len(_existing_plots)

    def _streamlit_safe_show(*args, **kwargs):
        global _plot_counter
        saved_paths = []
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            _plot_counter += 1
            plot_path = os.path.join(_plot_output_dir, f"plot_{_plot_counter}.png")
            fig.savefig(plot_path, bbox_inches="tight")
            saved_paths.append(plot_path)
        if saved_paths:
            print("Saved plot(s):")
            for path in saved_paths:
                print(f"- {path}")
        plt.close("all")

    plt.show = _streamlit_safe_show
except Exception:
    pass

import andes

# Load the 5-bus test case
ssa = andes.load(
    andes.get_case("5bus/pjm5bus.xlsx"),
    setup=True,
    no_output=True,
    log=False,
)

# Run the AC power flow
ssa.PFlow.run()