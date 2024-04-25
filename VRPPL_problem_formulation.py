##############Nodes ###################
#######Excel#######
# A) Dimension_X of the Node. For example N_0 = 0
# B) Dimension_Y of the Node. For example N_0 = 0
# C) Min_time window (a_i)
# D) Max_time window (b_i)
# E) Capacity of a locker station (ξ)
# F) Service time (s_i)
import pandas as pd
import pandas as pd
import pulp as lp
import math


node_excel = r"C:\Users\milto\Desktop\VRPPL_variables.xlsx"
#Read the values from the excel
df = pd.read_excel(node_excel)

##Example##
#Client = ("Home" or "Locker" or "Best_locker" or "Best_locker_or_home",
#         coordinates of where you want your package to go(x,y),min_arrival_time,max_arrival_time )
#Client = ("Locker", (30,15) )
Clients = [("Customer",(35,45) , 40, 5,9),
          ("Customer",(35,45) , 30, 7,9),
          ("Parcel",(90,100) , 35, 15,22),
          ("Customer",(70,48) , 40, 14,17),
          ("Customer",(67,32) , 50, 17,23),
          ("Customer",(70,80) , 90, 16,19),
          ("Parcel",(90,100) , 35, 19,22),
          ("Customer",(78,48) , 37, 2,8)
          
]

####This is the Nodes we already have. I you choose one of the following the the algorithm will be set to send the package in this coordinates
####If you insert a client of the second Comments the Costumer is in a random place and want to be delivered a package so the algorithm will search for the nearest node
"""
    ("Customer",(10,20) , 100, 5), ("Customer",(15,25) , 100, 10),("Customer",(5,15), 100, 8),
    ("Customer",(20,30) , 100, 12), ("Customer",(25,35) , 100, 7),("Customer",(30,40) , 100, 6),
    ("Customer",(35,45) , 100, 9), ("Customer",(40,50) , 100, 11), ("Customer",45,(55,100) , 13),
    ("Customer",(50,60) , 100, 14), ("Customer",(70,80) , 100, 18), ("Customer",(75,85) , 100, 19),
    ("Parcel",(80,90) , 57, 20), ("Parcel",(85,95) ,42, 21), ("Parcel",(90,100) , 35, 22),
    ("Parcel",(95,78) ,20, 23)
"""

#You can create any random node that you want those are just an example
"""
    ("Customer",(35,42) , 100, 5), ("Customer",(29,58) , 100, 10),("Customer",(89,25), 100, 8),
    ("Customer",(67,32) , 100, 12), ("Customer",(49,56) , 100, 7),("Customer",(70,48) , 100, 6)

"""


class Node():
    def __init__(self,Type:str,Dimention_x:int,Dimention_y:int,Capacity:int,Service_time:int):
        self.type = Type
        self.dim_x = Dimention_x
        self.dim_y = Dimention_y
        self.capacity = Capacity
        self.service_time = Service_time

    def to_dict(self):
        return {
            'Type': self.type,
            'Dimention_x': self.dim_x,
            'Dimention_y':self.dim_y,
            'Capacity': self.capacity,
            'Service_time': self.service_time
        }
    
    def add_the_node_to_excel(self, excel_path: str):
        # Read the existing data from the Excel file, or create a new DataFrame if the file does not exist
        try:
            df = pd.read_excel(excel_path)
        except FileNotFoundError:
            df = pd.DataFrame(columns=['Type', 'Node_coordinates', 'Capacity', 'Service_time'])

        # Convert the node to a dictionary and create a DataFrame from it
        node_data = pd.DataFrame([self.to_dict()])

        # Append the new node data to the existing DataFrame
        df = pd.concat([df, node_data], ignore_index=True)

        # Write the updated DataFrame back to the Excel file
        df.to_excel(excel_path, index=False)

def create_nodes(node_excel:str):
    #Node(Dimension_X,Dimension_Y,Capacity,Service_time)
    node0 = Node('Depot',0,0 , 100, 0)
    node0.add_the_node_to_excel(node_excel)

    nodes = [
    Node("Customer",10,20 , 100, 5), Node("Customer",15,25 , 100, 10), Node("Customer",5,15 , 100, 8),
    Node("Customer",20,30 , 100, 12), Node("Customer",25,35 , 100, 7), Node("Customer",30,40 , 100, 6),
    Node("Customer",35,45 , 100, 9), Node("Customer",40,50 , 100, 11), Node("Customer",45,55 , 100, 13),
    Node("Customer",50,60 , 100, 14), Node("Customer",55,65 , 100, 15), Node("Customer",60,70 , 100, 16),
    Node("Customer",65,75 , 100, 17), Node("Customer",70,80 , 100, 18), Node("Customer",75,85 , 100, 19),
    Node("Parcel",80,90 , 57, 20), Node("Parcel",85,95 ,42, 21), Node("Parcel",90,100 ,95, 105), 
    Node("Parcel",20,23, 34,3)
    ]

    # Adding each node to the Excel file
    for node in nodes:
        node.add_the_node_to_excel(node_excel)


class Customer():
    def __init__(self,client_type:str,client_coordinates:int,package_capacity:int, min_arrival_time:int, max_arrival_time:int):
        self.client_type = client_type # Unique identifier for the customer
        self.N_cc = client_coordinates# Includes all client coordinates
        self.package_capacity = package_capacity
        self.min_time = min_arrival_time  # Earliest delivery time window
        self.max_time = max_arrival_time  # Latest delivery time window

    def to_dict(self):
        return {
            "Client_type": self.client_type,
            "Client_coordinates": self.N_cc,
            "Package_capacity":self.package_capacity, 
            "min_time": self.min_time, 
            "max_time":self.max_time } 
         
            
def create_customer(client):
    # Unpack the client tuple
    client_type, client_coordinates,package_capacity, min_arrival_time, max_arrival_time = client

    # Validate client structure
    valid_types = {"Customer", "Parcel"}
    if (client_type in valid_types and 
        isinstance(client_coordinates, tuple) and len(client_coordinates) == 2 and
        all(isinstance(coord, int) for coord in client_coordinates) and
        0 < package_capacity < 100 and
        0 < min_arrival_time <= 100 and  # Assuming these are the valid ranges for times
        0 < max_arrival_time <= 100):

        # If validation passes, create and return the Customer object
        return Customer(client_type, client_coordinates, package_capacity,min_arrival_time, max_arrival_time)
    else:
        raise ValueError(f"Invalid client structure for client: {client}")



class Vehicle():
    def __init__(self, vehicles_available, vehicles_capacity):
        self.K = vehicles_available
        self.Q = vehicles_capacity
        # Initialize all vehicles at the depot location (0, 0)
        self.vehicles = [{"vehicle_id": i+1, "capacity": self.Q, "location": (0, 0)} for i in range(self.K)]
    
    def to_dict(self):
        return {
            "total_vehicles": self.K,
            "individual_vehicles": self.vehicles
        }
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def process_client_request(client,nodes):
    client_type, coordinates, _,_,_ = client
    dim_x = coordinates[0]
    dim_y = coordinates[1]
    if client_type == "Customer":
        for node in nodes:
            if dim_x == node.dim_x and dim_y == node.dim_y:
                return client
        return find_nearest_node(client,nodes)
    elif client_type == "Parcel":
        for node in nodes:
            if dim_x == node.dim_x and dim_y == node.dim_y:
                return client
        return find_nearest_node(client,nodes)
        
        # Process parcel-specific details

def find_nearest_node(client, nodes):
    nearest_node = None
    min_distance = float('inf')
    for node in nodes:
        distance = euclidean_distance(client[1], (node.dim_x, node.dim_y))
        if distance < min_distance:
            nearest_node = node
            min_distance = distance

    # Create a new tuple with updated coordinates
    updated_client = (client[0], (nearest_node.dim_x, nearest_node.dim_y), client[2], client[3], client[4])
    return updated_client

if __name__ == "__main__":
    M = 1000
    create_nodes(node_excel)
    nodes_df = pd.read_excel(node_excel)
    nodes = [Node(*node_info) for node_info in nodes_df.values.tolist()]

    processed_clients = []
    for client in Clients:
        processed_client = process_client_request(client, nodes)
        processed_clients.append(processed_client)

    customers = []
    for client in processed_clients:
        customers.append(create_customer(client))

    vehicle_fleet = Vehicle(5, 100)  # A fleet of 5 vehicles, each with a capacity of 100
    # Set the solver to Gurobi
    lp.LpSolverDefault = lp.GUROBI_CMD()

    model = lp.LpProblem("VRPPL_Optimization", lp.LpMinimize)
    nodes_df = nodes_df.values.tolist()
    # Decision variables: x[i][j][k] is 1 if vehicle k travels from node i to node j
    x = [[[lp.LpVariable(f'x_{i}_{j}_{k}', 0, 1, lp.LpBinary)
           for k in range(len(vehicle_fleet.vehicles))]
          for j in range(len(nodes_df))]
         for i in range(len(nodes_df))]
    
    arrival_times = [[lp.LpVariable(f'arrival_time_{i}_{k}', lowBound=0)
                  for k in range(len(vehicle_fleet.vehicles))]
                 for i in range(len(nodes_df))]

    # Objective: Minimize total distance traveled by all vehicles
    objective = lp.lpSum(euclidean_distance(nodes_df[i][1:3], nodes_df[j][1:3]) * x[i][j][k]
                         for i in range(len(nodes_df))
                         for j in range(len(nodes_df))
                         if i != j
                         for k in range(len(vehicle_fleet.vehicles)))
    model += objective

    # Constraints
    # Ensure each customer is visited exactly once
    for j in range(1, len(customers)):  # Assuming index 0 is the depot
        for k in range(len(vehicle_fleet.vehicles)):
            customer_index = j + 1
            model += arrival_times[customer_index][k] >= customers[j].min_time * x[0][j][k]
            model += arrival_times[customer_index][k] <= customers[j].max_time * x[0][j][k]

    for j in range(len(customers)):
        model += lp.lpSum(x[i][j][k]
                          for i in range(len(nodes_df))
                          for k in range(len(vehicle_fleet.vehicles))) == 1

    # Travel Time Constraints
    for i in range(len(nodes_df)):
        for j in range(1, len(nodes_df)):  # Assuming index 0 is the depot
            if i != j:
                for k in range(len(vehicle_fleet.vehicles)):
                    travel_time = euclidean_distance(nodes_df[i][1:3], nodes_df[j][1:3])
                    model += arrival_times[j][k] >= arrival_times[i][k] + travel_time - M * (1 - x[i][j][k])

    # Capacity constraint for each vehicle
    for k in range(len(vehicle_fleet.vehicles)):
         model += lp.lpSum(customers[j].package_capacity * x[i][j][k]
                      for i in range(len(nodes_df))
                      for j in range(len(customers))) <= vehicle_fleet.vehicles[k]['capacity']
    
    # Additional constraint to ensure the sum of package capacities does not exceed the vehicle's capacity
    for i in range(len(nodes_df)):
        model += lp.lpSum(customers[j].package_capacity * x[i][j][k]
                          for j in range(len(customers))) <= vehicle_fleet.vehicles[k]['capacity']

    # Depot constraint: Each vehicle starts from the depot (node 0) and leaves it only once
    for k in range(len(vehicle_fleet.vehicles)):
        model += lp.lpSum(x[0][j][k] for j in range(len(customers))) == 1

    # Solve the model
    model.solve()

    # Print solution
    if lp.LpStatus[model.status] == 'Optimal':
        total_distance = lp.value(objective)
        print(f"Total Distance Cost: {total_distance}")

        for k in range(len(vehicle_fleet.vehicles)):
            print(f"\nRoute for Vehicle {k + 1}:")
            route = []
            route_distance = 0.0
            capacity_used = 0.0
            prev_node = 0  # Starting from the depot (node 0)
            for i in range(len(nodes_df)):
                for j in range(len(nodes_df)):
                    if x[i][j][k].value() is not None and x[i][j][k].value() > 0.5:
                        route.append(f"Node {j}")
                        route_distance += euclidean_distance(nodes_df[prev_node][1:3], nodes_df[j][1:3])
                        prev_node = j
                        # Calculate capacity used
                        capacity_used += customers[j].package_capacity
            # Ensure the depot is included in the route
            route = ["Node 0"] + route
            print(" -> ".join(route))
            print(f"Route Distance for Vehicle {k + 1}: {route_distance}")
            print(f"Capacity of Vehicle {k + 1}: {vehicle_fleet.vehicles[k]['capacity']}")
            print(f"Capacity Left on Vehicle {k + 1}: {vehicle_fleet.vehicles[k]['capacity'] - capacity_used}")
    else:
        print("No optimal solution found")


    