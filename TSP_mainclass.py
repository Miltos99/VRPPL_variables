from itertools import permutations
from algorithms import *
import math

class BaseVertex:
    def __init__(self, id, x, y, truck_speed, drone_speed,name="", is_depot=False):
        self.id = id
        self.x = x
        self.y = y
        self.name = name
        self.truck_speed = truck_speed
        self.drone_speed = drone_speed
        self.is_depot = is_depot

    def distance_to(self, other):
        distance = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        return distance

    def truck_speed(self):
        return self.truck_speed

    def drone_speed(self):
        return self.drone_speed

class TSP:
    def __init__(self):
        self.truck_speed = 0
        self.drone_speed = 0
        self.original_list = []

    def read_file(self, filename):
        with open(filename, 'r') as file:
            print("Loading File...")
            lines = file.readlines()

            # Extract speeds
            truck_speed= lines[lines.index("/*The speed of the Truck*/\n") + 1].strip()
            self.truck_speed = float(truck_speed)

            drone_speed = lines[lines.index("/*The speed of the Drone*/\n") + 1].strip()
            self.drone_speed = float(drone_speed)


            # Extract locations
            locations_start = lines.index("/*The Locations (x_coor y_coor name)*/\n") + 1
            for line in lines[locations_start:]:
                if line.strip():
                    x, y= line.strip().split()[:2]
                    self.add_vertex(float(x), float(y), drone_speed, truck_speed)
        
    def read_solution_file(self, filename):
        data_lines = []
        with open(filename, 'r') as file:
            for i, line in enumerate(file, start=1):
                if i >= 5:
                    data_lines.append(line.strip()) 
        data_lines = [int(line.split('\t')[0]) for line in data_lines]
        data_lines.append(0)
        return data_lines
        

    def add_vertex(self, x, y, truck_speed,drone_speed, is_depot=False):
        self.original_list.append(BaseVertex(len(self.original_list), x, y, truck_speed,drone_speed,is_depot))


    def All_algorithms(self, is_random = bool,route_list = []):
        algorithm = algorithms(self.original_list,self.truck_speed,self.drone_speed)
        #shortest_tour, min_distance = algorithm.brute_force_only_truck()
        #shortest_tour, min_distance  = algorithm.nearest_neighbor()
        #shortest_tour, min_distance = algorithm.two_opt()
        
        if is_random:
            optimal_solution = []
            for _ in range(400):
                #shortest_tour, min_distance, route_list = algorithm.drone_optimization_if_feasible_serve(True)
                shortest_tour, min_distance, route_list = algorithm.drone_forward_and_backward_search(True)
                optimal_solution.append((shortest_tour, min_distance, route_list))
            
            # Select the optimal solution with the smallest min_distance
            shortest_tour, min_distance, route_list = min(optimal_solution, key=lambda x: x[1])

        else:
            #shortest_tour, min_distance, route_list = algorithm.drone_optimization_if_feasible_serve()
            shortest_tour, min_distance, route_list = algorithm.drone_forward_and_backward_search()
        
    
        
        test_drone_and_truck, test_min_distance = algorithm.test_algorithm(route_list)
        return shortest_tour, min_distance,test_drone_and_truck, test_min_distance
    





       
