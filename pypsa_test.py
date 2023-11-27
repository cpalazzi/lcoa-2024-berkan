import pypsa
import gurobipy as grb
import pandas as pd
import pyomo.environ as pyo

def add_ramping_constraints(net, snapshots):
    m = net.model

    # Ramping limits (change these values as needed)
    ramping_up_limit = 20  # MW/hour
    ramping_down_limit = 20  # MW/hour

    for gen in net.generators.index:
        if gen in m.Generators.p:
            m.Ramping_Up_Limit.add(net.generators.loc[gen, "p_max_pu"] - m.Generators.p[gen, 0] <= ramping_up_limit)
            m.Ramping_Down_Limit.add(m.Generators.p[gen, 0] - net.generators.loc[gen, "p_min_pu"] <= ramping_down_limit)

# Create a PyPSA network
network = pypsa.Network()

# Add a bus and a generator
network.add("Bus", "bus")
network.add("Generator", "gen", bus="bus", p_nom=100, p_max_pu=1.0, p_min_pu=0.0)

# Add a load
network.add("Load", "load", bus="bus", p_set=50)

# Add generation costs
network.generators.loc["gen", "marginal_cost"] = 10.0  # Set a cost of 10 USD/MWh

# Set the objective to minimize the generation costs
network.optimize(solver_name="gurobi", extra_functionality=add_ramping_constraints)

# Get the objective value (total generation costs)
objective_value = network.objective

# Print the objective value
print(f"Objective Value (Total Generation Costs): {objective_value}")

# Save the objective value to a CSV file using Pandas
df = pd.DataFrame({"Objective": [objective_value]})
df.to_csv("objective_pypsa.csv", index=False)
