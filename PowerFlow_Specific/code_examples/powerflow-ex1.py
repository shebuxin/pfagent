import andes
import os

script_dir = os.getcwd()
case = os.path.join(script_dir, 'ieee39_base.xlsx')

ssa = andes.load(case,
                setup=False,
                no_output=True,
                default_config=False)

# setup the case and call the power flow calculation
ssa.setup()
ssa.PFlow.run()

print("PV Bus Active Power", ssa.PV.p.v)
print("PV Bus Reactive Power", ssa.PV.q.v)
print("Slack Bus Active Power", ssa.Slack.p.v)
print("Slack Bus Reactive Power", ssa.Slack.q.v)
print("Bus Voltages", ssa.Bus.v.v)
print("Bus Angles", ssa.Bus.a.v)

print("Phase Angle of the From Bus", ssa.Line.a1.e)
print("Shape of ssa.Line.a1.e:", ssa.Line.a1.e.shape)
print("ssa.Line.a1.e via get():", ssa.Line.get(src='a1', idx= ssa.Line.idx.v, attr='e'))
print("Phase Angle of the To Bus", ssa.Line.a2.e)


# Run a power-flow simulation and display key results for the PV buses, Slack bus, system buses, and transmission lines.