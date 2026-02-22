# required_dependencies: pandapower
import pandapower as pp

# Create an empty power system
net = pp.create_empty_network()

# Add buses to the network
bus1 = pp.create_bus(net, vn_kv=20, name="Bus 1")
bus2 = pp.create_bus(net, vn_kv=0.4, name="Bus 2")

# Create a transformer between the buses
pp.create_transformer(net, bus1, bus2, std_type="0.4 MVA 20/0.4 kV")

# Add a load at Bus 2
pp.create_load(net, bus=bus2, p_mw=0.1, q_mvar=0.05)

# Add a generator at Bus 1
pp.create_gen(net, bus=bus1, p_mw=0.2, min_q_mvar=-0.1, max_q_mvar=0.1)

# Perform the power flow calculation
pp.runpp(net)

# Output the results of the power flow
print(net.res_bus)
print(net.res_line)