import andes
import andes.kpi

# Load the full Kundur case (including dynamics) in ANDES
andes = andes.load(
    andes.get_case('kundur/kundur_full.xlsx'),
    setup=False, no_output=True, actions=False
)
andes.setup()
andes.PFlow.run()

# Compute and print various KPI metrics
kpi_metrics = {
    'KPI_PV_P_MEAN': andes.kpi.PV_P_MEAN.v,
    'KPI_PV_Q_MEAN': andes.kpi.PV_Q_MEAN.v,
    'KPI_Slack_P_SUM': andes.kpi.Slack_P_SUM.v,
    'KPI_Slack_Q_SUM': andes.kpi.Slack_Q_SUM.v,
    'KPI_V_MIN': andes.kpi.V_MIN.v,
    'KPI_V_MAX': andes.kpi.V_MAX.v,
}

for key, value in kpi_metrics.items():
    print(f'{key} = {value}')