import random
import time as tm
import numpy as np
from TSP_mainclass import *
filepath = "C:/Users/milto/Desktop/Github/TCP_D/data.txt"
test_filepath = "C:/Users/milto/Desktop/Github/TCP_D/test_data.txt"


def validate_tour(shortest_tour, drone_deliveries=[]):
    total_nodes = len(shortest_tour) - 1 
    if shortest_tour.count(0) != 2:
        raise ValueError("The depot node (0) does not appear exactly twice in the tour.")


    launching_nodes = [delivery[0] for delivery in drone_deliveries]
    delivery_nodes = [delivery[1] for delivery in drone_deliveries]
    meeting_nodes = [delivery[2] for delivery in drone_deliveries]


    if len(delivery_nodes) != len(set(delivery_nodes)):
        raise ValueError("A delivery is happening on the same node more than once.")


    for node in delivery_nodes:
        if node in shortest_tour:
            raise ValueError(f"Both truck and drone are delivering to the same node {node}.")


    if len(launching_nodes) != len(set(launching_nodes)):
        raise ValueError("Duplicate launching nodes detected.")
    if len(meeting_nodes) != len(set(meeting_nodes)):
        raise ValueError("Duplicate meeting nodes detected.")


    combined_nodes = shortest_tour + delivery_nodes
    combined_nodes_set = set(combined_nodes)
    missing_nodes = set(range(total_nodes)) - combined_nodes_set
    if missing_nodes:
        raise ValueError(f"Missing node(s): {', '.join(map(str, missing_nodes))}")


    if any(node == 0 for node in delivery_nodes):
        raise ValueError("Invalid drone delivery involving the depot.")  
    

if __name__ == "__main__":
    start_time = tm.time()
    tsp = TSP()
    #shortest_tour, min_distance = tsp.brute_force_only_truck()
    tsp.read_file(filepath)
    #route_list = tsp.read_solution_file(test_filepath)
    #test_drone_and_truck, test_min_distance  = tsp.All_algorithms(True, route_list)
    #test_truck_tour, test_drone_tour = test_drone_and_truck

    drone_and_truck, min_distance,test_drone_and_truck, test_min_distance  = tsp.All_algorithms(False)
    truck_tour, drone_tour = drone_and_truck
    try:
        print(f"The total distance is: {test_min_distance}")
        print(f"The minimum distance is: {min_distance}") 
        print(f"The total saving is: {test_min_distance - min_distance}")
        print(f"Percentage reduced: {((test_min_distance - min_distance) /test_min_distance)*100} %")
        #validate_tour(test_truck_tour,test_drone_tour)
    except ValueError as e:
        print(f"Error in tour: {e}")

    end_time = tm.time()
    elapsed_time = end_time - start_time

   # print(f"The code took {elapsed_time} seconds to run.")