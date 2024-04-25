from itertools import permutations
from pulp import *
import pulp
import copy
import random



class algorithms():
    def __init__(self, original_list,truck_speed,drone_speed,battery_life= 30, battery_swapping_time = 0):
        self.original_list = original_list
        self.distance_matrix = self.create_distance_matrix(self.original_list)
        self.truck_speed = truck_speed
        self.drone_speed = drone_speed
        self.battery_life = battery_life
        self.battery_swapping_time =  battery_swapping_time
    
    def create_distance_matrix(self, original_list):
        n = len(original_list)
        distance_matrix = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                
                distance_matrix[i][j] = self.original_list[i].distance_to(original_list[j])
        return distance_matrix

    def calculate_distance(self, tour):
        return sum(self.distance_matrix[tour[i]][tour[(i+1) % len(tour)]] for i in range(len(tour)))

    def two_opt_swap(self, route, i, k):
        """Take route[0] to route[i-1] and add route[k] to route[i] in reverse order, then add the rest."""
        return route[0:i] + route[k:i-1:-1] + route[k+1:]

    def calculate_partial_distance(self,truck_route, launch_index, meetup_index):
        return self.distance_matrix[truck_route[launch_index]][truck_route[meetup_index]]




######################## TRUCK_ALGORITHMS ########################################
    def brute_force_only_truck(self):
        distance_matrix = self.create_distance_matrix(self.original_list)
        shortest_tour = None
        min_distance = float('inf')
        
        for perm in permutations(range(1, len(distance_matrix))):
            tour = (0,) + perm + (0,) 
            current_distance = sum(distance_matrix[tour[i]][tour[i+1]] for i in range(len(tour) - 1))
            if current_distance < min_distance:
                min_distance = current_distance
                shortest_tour = tour
                
        return shortest_tour, min_distance
    
    


    def nearest_neighbor(self):
        distance_matrix = self.create_distance_matrix(self.original_list)
    # Initialize variables
        n = len(distance_matrix)
        visited_nodes = [False] * n
        route = [0]  # Start from the depot node
        total_distance = 0

        current_node = 0
        visited_nodes[current_node] = True

        for _ in range(1, n):
            next_node = None
            shortest_distance = float('inf')
            for j in range(1, n):
                if not visited_nodes[j] and distance_matrix[current_node][j] < shortest_distance:
                    next_node = j
                    shortest_distance = distance_matrix[current_node][j]

            if next_node is not None:
                route.append(next_node)
                total_distance += shortest_distance
                current_node = next_node
                visited_nodes[current_node] = True

        total_distance += distance_matrix[current_node][0]
        route.append(0)

        return route, total_distance
    

    def two_opt(self):
        best_final_distance = float('inf')
        best_final_route = None
        for start_node in range(len(self.distance_matrix)):
            route = list(range(len(self.distance_matrix)))
            route = route[start_node:] + route[:start_node]  # Rotate route to start from start_node
            route.append(route[0])  # Include initial node at the end
            best_distance = self.calculate_distance(route)  # Calculate initial distance
            val = len(route)
            improvement = True
            while val >= 0:
                while improvement:
                    constrain = True
                    improvement = False
                    for i in range(1, len(route) - val):
                        for k in range(i + val, len(route)):
                            new_route = self.two_opt_swap(route, i, k)
                            new_distance = self.calculate_distance(new_route)
                            if new_distance < best_distance and constrain:
                                route = new_route
                                best_distance = new_distance
                                constrain = False
                                improvement = True
                    if not improvement:
                        val -= 1
                        improvement = True
                    if val < 0:
                        improvement = False
                if best_distance < best_final_distance:
                    best_final_distance = best_distance
                    best_final_route = route
                val -= 1
        return best_final_route, best_final_distance
    
    def random_route(self):
        # The self.distance_matrix its length gives the number of nodes
        node_count = len(self.distance_matrix)
        # Creating a list of nodes based on the number of nodes
        route = list(range(node_count))
        # Shuffling the list of nodes to create a random route
        random.shuffle(route)
        return route

    def test_algorithm(self,route_list):
        distance_matrix = self.create_distance_matrix(self.original_list)
        total_distance = 0
        for i in range(len(route_list) - 1):
            total_distance += distance_matrix[route_list[i]][route_list[i+1]]
        return [route_list,[]],total_distance
    


############################ DRONE_ALGORITHM ###########################
    def drone_forward_and_backward_search(self, is_random=False): 
        if is_random:
            truck_route = self.random_route()
        else:
            truck_route = self.two_opt()[0]

        rev_truck_route = list(reversed(truck_route))
        optimized_truck_route = truck_route.copy()
        
        # Function to calculate savings
        def calculate_savings(truck_time_without_drone, truck_time_bypassing_delivery):
            return truck_time_without_drone - truck_time_bypassing_delivery
        
        # Function to find optimal deliveries
        def find_optimal_deliveries(optimized_truck_route):
            next_saving_time = 0
            saving_time = 0
            drone_deliveries = []
            launch_index = 0
            print(optimized_truck_route)
            while launch_index < len(optimized_truck_route) - 2:
                delivery_made = False
                for delivery_index in range(launch_index + 1, launch_index + 2):
                    for meetup_index in range(delivery_index + 1, len(optimized_truck_route)):
                        is_feasible, truck_time_bypassing_delivery, truck_route_bypassing_delivery,truck_time_without_drone, truck_time_without_drone_ = self.is_drone_delivery_feasible(optimized_truck_route, launch_index, delivery_index, meetup_index)
                        if is_feasible:
                            drone_delivery = []
                            print(f"truck_tour_bypassing_delivery { truck_route_bypassing_delivery}")
                            print(f"truck_tour_without_drone{truck_time_without_drone_}")
                            print(f"saving_time {truck_time_without_drone - truck_time_bypassing_delivery}")
                            saving_time = calculate_savings(truck_time_without_drone,truck_time_bypassing_delivery)
                            drone_delivery.append([launch_index, delivery_index, meetup_index])
                            launch_index += 1
                            delivery_made = True
                            break 
                    if delivery_made:
                        break 

                if not delivery_made:
                    launch_index += 1 
                
                if delivery_made:
                    delivery_made = False
                    for delivery_index in range(launch_index + 1, launch_index + 2):
                        for meetup_index in range(delivery_index + 1, len(optimized_truck_route)):
                            is_feasible, next_truck_time_bypassing_delivery, next_truck_route_bypassing_delivery, next_truck_time_without_drone, next_truck_time_without_drone_ =\
                                self.is_drone_delivery_feasible(optimized_truck_route, launch_index, delivery_index, meetup_index)
                            if is_feasible:
                                print(f"next_truck_tour_bypassing_delivery { next_truck_route_bypassing_delivery}")
                                print(f"next_truck_tour_without_drone{next_truck_time_without_drone_}")
                                print(f"next_saving_time {next_truck_time_without_drone - next_truck_time_bypassing_delivery}")
                                next_saving_time = calculate_savings(next_truck_time_without_drone, next_truck_time_bypassing_delivery)
                                
                                if saving_time >= next_saving_time:
                                    drone_deliveries.append((optimized_truck_route[drone_delivery[0][0]],optimized_truck_route[drone_delivery[0][1]],optimized_truck_route[drone_delivery[0][2]],saving_time))
                                    optimized_truck_route.pop(drone_delivery[0][1])
                                    launch_index = drone_delivery[0][2] - 1
                                    break
                                else:
                                    drone_deliveries.append((optimized_truck_route[launch_index], optimized_truck_route[delivery_index], optimized_truck_route[meetup_index],next_saving_time))
                                    optimized_truck_route.pop(delivery_index) 
                                    launch_index = meetup_index - 1
                                delivery_made = True
                                break 
                        if delivery_made:
                            break 

                    if not delivery_made:
                        launch_index += 1

            return drone_deliveries, optimized_truck_route

        # Find optimal deliveries for both forward and reverse routes
        forward_deliveries, forwardTrucktour = find_optimal_deliveries(optimized_truck_route)
        backward_deliveries,backwardTrucktour = find_optimal_deliveries(rev_truck_route)

        # Compare savings and choose the route with higher savings
        forward_savings = sum(delivery[3] for delivery in forward_deliveries)
        backward_savings = sum(delivery[3] for delivery in backward_deliveries)

        if forward_savings >= backward_savings:
            final_route = forwardTrucktour
            drone_deliveries = forward_deliveries
        else:
            final_route = backwardTrucktour
            drone_deliveries = backward_deliveries

        final_distance = self.calculate_distance(final_route)
        if not drone_deliveries:
            print("No feasible drone delivery found.")
        
        print(f"The battery swapping time is = {self.battery_swapping_time}")
        
        return [final_route, drone_deliveries], final_distance, truck_route

    def drone_optimization_if_feasible_serve(self, is_random=False):
        
        #If is_random =True then random truck route
        if is_random:
            truck_route = self.random_route()
        else: #If is_random = False then the truck route is been calculated from two_opt()
            truck_route = self.two_opt()[0]


        optimized_truck_route = truck_route.copy()

        #With the next instruction, the truck will reverse the route.
        #optimized_truck_route = list(reversed(optimized_truck_route))

        drone_deliveries = []
        launch_index = 0
        print(optimized_truck_route)
        while launch_index < len(optimized_truck_route) - 2:
            delivery_made = False
            for delivery_index in range(launch_index + 1, launch_index + 2):
                for meetup_index in range(delivery_index + 1, len(optimized_truck_route)):
                    is_feasible, truck_time_bypassing_delivery, truck_route_bypassing_delivery,truck_time_without_drone, truck_time_without_drone_ = self.is_drone_delivery_feasible(optimized_truck_route, launch_index, delivery_index, meetup_index)
                    if is_feasible:
                        print(f"truck_tour_bypassing_delivery { truck_route_bypassing_delivery}")
                        print(f"truck_tour_without_drone{truck_time_without_drone_}")
                        print(f"saving_time {truck_time_without_drone - truck_time_bypassing_delivery}")
                        drone_deliveries.append((optimized_truck_route[launch_index], optimized_truck_route[delivery_index], optimized_truck_route[meetup_index]))
                        optimized_truck_route.pop(delivery_index) 
                        launch_index = meetup_index - 1
                        delivery_made = True
                        break 
                if delivery_made:
                    break 

            if not delivery_made:
                launch_index += 1 

        final_distance = self.calculate_distance(optimized_truck_route)

        if not drone_deliveries:
            print("No feasible drone delivery found.")
        print(f"The battery swapping time is = {self.battery_swapping_time}")
        return [optimized_truck_route, drone_deliveries], final_distance, truck_route

    def is_drone_delivery_feasible(self, optimized_truck_route, launch_index, delivery_index, meetup_index):
        launch_point = optimized_truck_route[launch_index]
        delivery_point = optimized_truck_route[delivery_index]
        meetup_point = optimized_truck_route[meetup_index]

        to_delivery_distance = self.distance_matrix[launch_point][delivery_point]
        to_meetup_distance = self.distance_matrix[delivery_point][meetup_point]
        drone_delivery_time = (to_delivery_distance + to_meetup_distance) / self.drone_speed

        if drone_delivery_time <= self.battery_life:
            # Adjusted: Calculate truck route time bypassing the delivery node
            truck_route_bypassing_delivery = optimized_truck_route[launch_index:delivery_index] + optimized_truck_route[delivery_index + 1:meetup_index + 1]
            truck_time_bypassing_delivery = self.calculate_distance(truck_route_bypassing_delivery) / self.truck_speed + self.battery_swapping_time
            truck_time_without_drone_ = optimized_truck_route[launch_index:meetup_index+1]
            truck_time_without_drone = self.calculate_distance(truck_time_without_drone_) / self.truck_speed
            total_truck_time_without_drone = self.calculate_distance(optimized_truck_route[launch_index:meetup_index + 1]) / self.truck_speed
            is_truck_soone = truck_time_bypassing_delivery > drone_delivery_time and self.calculate_distance(truck_route_bypassing_delivery) / self.truck_speed <= self.battery_life
            
            is_feasible = total_truck_time_without_drone > truck_time_bypassing_delivery and is_truck_soone
            return is_feasible, truck_time_bypassing_delivery, truck_route_bypassing_delivery,truck_time_without_drone, truck_time_without_drone_

        return False,0,0,0,0
