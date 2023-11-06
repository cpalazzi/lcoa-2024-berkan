import pypsa
import gurobipy as grb
import pandas as pd

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
network.lopf(network.snapshots, solver_name="gurobi")

# Get the objective value (total generation costs)
objective_value = network.objective

# Print the objective value
print(f"Objective Value (Total Generation Costs): {objective_value}")

# Save the objective value to a CSV file using Pandas
df = pd.DataFrame({"Objective": [objective_value]})
df.to_csv("objective_pypsa.csv", index=False)
