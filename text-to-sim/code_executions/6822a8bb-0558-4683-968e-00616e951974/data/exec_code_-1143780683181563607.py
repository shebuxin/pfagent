# required_dependencies: pandapower
import pandapower as pp

# Create an empty power system
net = pp.create_empty_network()

# Create buses
bus1 = pp.create_bus(net, vn_kv=20, name="Bus 1")
bus2 = pp.create_bus(net, vn_kv=0.4, name="Bus 2")

# Create a transformer between the buses
pp.create_transformer(net, bus1, bus2, std_type="0.4 MVA 20/0.4 kV")

# Create a load at Bus 2
pp.create_load(net, bus=bus2, p_mw=0.1, q_mvar=0.05)

# Create a generator at Bus 1
pp.create_gen(net, bus=bus1, p_mw=0.2, min_q_mvar=-0.1, max_q_mvar=0.1)

# Run the power flow calculation
pp.runpp(net)

# Print results
print(net.res_bus)
print(net.res_line)