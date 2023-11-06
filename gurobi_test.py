import gurobipy as grb
import pandas as pd

# Create a Gurobi model
model = grb.Model()

# Create decision variables
x = model.addVar(vtype=grb.GRB.CONTINUOUS, name="x")
y = model.addVar(vtype=grb.GRB.CONTINUOUS, name="y")

# Set objective function: Maximize x + y
model.setObjective(x + y, grb.GRB.MAXIMIZE)

# Add constraints
model.addConstr(x + 2 * y <= 10, name="c1")
model.addConstr(3 * x - y <= 12, name="c2")

# Optimize the model
model.optimize()

# Check optimization status
if model.status == grb.GRB.OPTIMAL:
    # Print the objective value
    objective_value = model.objVal
    print(f"Objective Value: {objective_value}")

    # Save the objective value to a CSV file using Pandas
    df = pd.DataFrame({"Objective": [objective_value]})
    df.to_csv("objective.csv", index=False)
else:
    print("Optimization did not reach the optimal solution.")

# You can access variable values using x.x and y.x
# x_value = x.x
# y_value = y.x
