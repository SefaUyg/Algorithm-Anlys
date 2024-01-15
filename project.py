import random
import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class Node:
    def __init__(self, id):
        self.id = id
        self.neighbors = []
        self.distances = {}
        self.bandwidths = {}
        self.delays = {}
        self.reliabilities = {}

class Edge:
    def __init__(self, node1, node2, distance, bandwidth, delay, reliability):
        self.node1 = node1
        self.node2 = node2
        self.distance = distance
        self.bandwidth = bandwidth
        self.delay = delay
        self.reliability = reliability

def read_input_file(filename):
    nodes = []
    edges = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Read adjacency matrix
    adjacency_matrix = []
    for line in lines:
        if line.strip():
            adjacency_matrix.append([int(x) for x in line.split(':')])

    # Read bandwidth matrix
    bandwidth_matrix = []
    for line in lines:
        if line.strip():
            bandwidth_matrix.append([int(x) for x in line.split(':')])

    # Read delay matrix
    delay_matrix = []
    for line in lines:
        if line.strip():
            delay_matrix.append([int(x) for x in line.split(':')])

    # Read reliability matrix
    reliability_matrix = []
    for line in lines:
        if line.strip():
            reliability_matrix.append([float(x) for x in line.split(':')])

    # Create nodes
    for i in range(len(adjacency_matrix)):
        node = Node(i)
        nodes.append(node)

    # Create edges
    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix)):
            if adjacency_matrix[i][j] == 1:
                distance = random.randint(1, 5)
                bandwidth = random.randint(3, 10)
                delay = random.randint(1, 5)
                reliability = random.uniform(0.95, 0.99)
                edge = Edge(nodes[i], nodes[j], distance, bandwidth, delay, reliability)
                edges.append(edge)
                nodes[i].neighbors.append(nodes[j])
                nodes[i].distances[nodes[j]] = distance
                nodes[i].bandwidths[nodes[j]] = bandwidth
                nodes[i].delays[nodes[j]] = delay
                nodes[i].reliabilities[nodes[j]] = reliability

    return nodes, edges

def dijkstra(nodes, edges, source, destination, bandwidth_demand):
    # Initialize distances and previous nodes
    distances = {node: float('inf') for node in nodes}
    distances[source] = 0
    previous = {node: None for node in nodes}

    # Initialize priority queue
    pq = [(0, source)]

    # While there are still nodes to visit
    while pq:
        # Get the node with the smallest distance
        current_distance, current_node = heapq.heappop(pq)
        # If we have reached the destination, we are done
        if current_node == destination:
            break

        # For each neighbor of the current node
        for neighbor in current_node.neighbors:
            # Calculate the new distance to the neighbor
            new_distance = current_distance + current_node.distances[neighbor]
            # If the new distance is shorter than the current distance to the neighbor
            if new_distance < distances[neighbor]:
                # Update the distance and previous node
                distances[neighbor] = new_distance
                previous[neighbor] = current_node
                # Add the neighbor to the priority queue
                heapq.heappush(pq, (new_distance, neighbor))

    # Reconstruct the path from the destination to the source
    path = []
    current_node = destination
    while current_node is not None:
        path.append(current_node)
        current_node = previous[current_node]

    # Reverse the path to get the path from the source to the destination
    path.reverse()

    # Check if the path satisfies the bandwidth demand
    if not check_bandwidth_demand(path, bandwidth_demand):
        return None

    return path

def check_bandwidth_demand(path, bandwidth_demand):
    # Check if the bandwidth of each edge in the path is greater than or equal to the bandwidth demand
    for i in range(len(path) - 1):
        if path[i].bandwidths[path[i + 1]] < bandwidth_demand:
            return False

    return True

def bellman_ford(nodes, edges, source, destination, bandwidth_demand):
    # Initialize distances and previous nodes
    distances = {node: float('inf') for node in nodes}
    distances[source] = 0
    previous = {node: None for node in nodes}

    # Relax all edges |V| - 1 times
    for _ in range(len(nodes) - 1):
        for edge in edges:
            # If the new distance to the neighbor is shorter than the current distance to the neighbor
            if distances[edge.node1] + edge.distance < distances[edge.node2]:
                # Update the distance and previous node
                distances[edge.node2] = distances[edge.node1] + edge.distance
                previous[edge.node2] = edge.node1

    # Check for negative weight cycles
    for edge in edges:
        if distances[edge.node1] + edge.distance < distances[edge.node2]:
            # Negative weight cycle found, return None
            return None

    # Reconstruct the path from the destination to the source
    path = []
    current_node = destination
    while current_node is not None:
        path.append(current_node)
        current_node = previous[current_node]

    # Reverse the path to get the path from the source to the destination
    path.reverse()

    # Check if the path satisfies the bandwidth demand
    if not check_bandwidth_demand(path, bandwidth_demand):
        return None

    return path



def a_star(nodes, edges, source, destination, bandwidth_demand):
    # Initialize open and closed sets
    open_set = [source]
    closed_set = set()

    # Initialize g_scores and h_scores
    g_scores = {node: float('inf') for node in nodes}
    g_scores[source] = 0
    h_scores = {node: math.inf for node in nodes}
    h_scores[destination] = 0

    # Initialize f_scores
    f_scores = {node: float('inf') for node in nodes}
    f_scores[source] = h_scores[source]

    # While the open set is not empty
    while open_set:
        # Get the node with the lowest f_score
        current_node = min(open_set, key=lambda node: f_scores[node])

        # If we have reached the destination, we are done
        if current_node == destination:
            break

        # Move the current node from the open set to the closed set
        open_set.remove(current_node)
        closed_set.add(current_node)

        # For each neighbor of the current node
        for neighbor in current_node.neighbors:
            # If the neighbor is in the closed set, skip it
            if neighbor in closed_set:
                continue

            # Calculate the new distance, delay sum, and f_score
            new_distance = current_node.distances[neighbor]
            new_delay_sum = delay_sums[current_node] + current_node.delays[neighbor]
            new_f_score = g_scores[current_node] + new_distance + h_scores[neighbor]

            # If the new distance and delay sum are shorter than the current distance and delay sum to the neighbor
            if new_distance < distances[neighbor] and new_delay_sum < delay_threshold:
                # Update the distance, previous node, delay sum, and f_score
                distances[neighbor] = new_distance
                previous[neighbor] = current_node
                delay_sums[neighbor] = new_delay_sum
                f_scores[neighbor] = new_f_score

                # If the neighbor is not in the open set, add it
                if neighbor not in open_set:
                    open_set.append(neighbor)

    # Reconstruct the path from the destination to the source
    path = []
    current_node = destination
    while current_node is not None:
        path.append(current_node)
        current_node = previous[current_node]

    # Reverse the path to get the path from the source to the destination
    path.reverse()

    # Check if the path satisfies the bandwidth demand
    if not check_bandwidth_demand(path, bandwidth_demand):
        return None

    return path

def dijkstra_with_delay_constraint(nodes, edges, source, destination, bandwidth_demand, delay_threshold):
    # Initialize distances, previous nodes, and delay sums
    distances = {node: float('inf') for node in nodes}
    distances[source] = 0
    previous = {node: None for node in nodes}
    delay_sums = {node: 0 for node in nodes}

    # Initialize priority queue
    pq = [(0, 0, source)]

    # While there are still nodes to visit
    while pq:
        # Get the node with the smallest distance and delay sum
        current_distance, current_delay_sum, current_node = heapq.heappop(pq)

        # If we have reached the destination, we are done
        if current_node == destination:
            break

        # For each neighbor of the current node
        for neighbor in current_node.neighbors:
            # Calculate the new distance, delay sum, and f_score
            new_distance = current_node.distances[neighbor]
            new_delay_sum = delay_sums[current_node] + current_node.delays[neighbor]
            new_f_score = g_scores[current_node] + new_distance + h_scores[neighbor]

            # If the new distance and delay sum are shorter than the current distance and delay sum to the neighbor
            if new_distance < distances[neighbor] and new_delay_sum < delay_threshold:
                # Update the distance, previous node, delay sum, and f_score
                distances[neighbor] = new_distance
                previous[neighbor] = current_node
                delay_sums[neighbor] = new_delay_sum
                f_scores[neighbor] = new_f_score

                # If the neighbor is not in the open set, add it
                if neighbor not in open_set:
                    open_set.append(neighbor)

    # Reconstruct the path from the destination to the source
    path = []
    current_node = destination
    while current_node is not None:
        path.append(current_node)
        current_node = previous[current_node]

    # Reverse the path to get the path from the source to the destination
    path.reverse()

    # Check if the path satisfies the bandwidth demand
    if not check_bandwidth_demand(path, bandwidth_demand):
        return None

    return path

def bellman_ford_with_delay_constraint(nodes, edges, source, destination, bandwidth_demand, delay_threshold):
    # Initialize distances, previous nodes, and delay sums
    distances = {node: float('inf') for node in nodes}
    distances[source] = 0
    previous = {node: None for node in nodes}
    delay_sums = {node: 0 for node in nodes}

    # Relax all edges |V| - 1 times
    for _ in range(len(nodes) - 1):
        for edge in edges:
            # If the new distance and delay sum to the neighbor are shorter than the current distance and delay sum to the neighbor
            if distances[edge.node1] + edge.distance < distances[edge.node2] and delay_sums[edge.node1] + edge.delay < delay_threshold:
                # Update the distance, previous node, delay sum, and f_score
                distances[edge.node2] = distances[edge.node1] + edge.distance
                previous[edge.node2] = edge.node1
                delay_sums[edge.node2] = delay_sums[edge.node1] + edge.delay

    # Check for negative weight cycles
    for edge in edges:
        if distances[edge.node1] + edge.distance < distances[edge.node2] and delay_sums[edge.node1] + edge.delay < delay_threshold:
            # Negative weight cycle found, return None
            return None

    # Reconstruct the path from the destination to the source
    path = []
    current_node = destination
    while current_node is not None:
        path.append(current_node)
        current_node = previous[current_node]

    # Reverse the path to get the path from the source to the destination
    path.reverse()

    # Check if the path satisfies the bandwidth demand
    if not check_bandwidth_demand(path, bandwidth_demand):
        return None

    return path

def a_star_with_delay_constraint(nodes, edges, source, destination, bandwidth_demand, delay_threshold):
    # Initialize open and closed sets
    open_set = [source]
    closed_set = set()

    # Initialize g_scores, h_scores, and delay sums
    g_scores = {node: float('inf') for node in nodes}
    g_scores[source] = 0
    h_scores = {node: math.inf for node in nodes}
    h_scores[destination] = 0
    delay_sums = {node: 0 for node in nodes}

    # Initialize f_scores
    f_scores = {node: float('inf') for node in nodes}
    f_scores[source] = h_scores[source]

    # While the open set is not empty
    while open_set:
        # Get the node with the lowest f_score
        current_node = min(open_set, key=lambda node: f_scores[node])

        # If we have reached the destination, we are done
        if current_node == destination:
            break

        # Move

        # Move the current node from the open set to the closed set
        open_set.remove(current_node)
        closed_set.add(current_node)

        # For each neighbor of the current node
        for neighbor in current_node.neighbors:
            # If the neighbor is in the closed set, skip it
            if neighbor in closed_set:
                continue

            # Calculate the new distance, delay sum, and f_score
            new_distance = current_node.distances[neighbor]
            new_delay_sum = delay_sums[current_node] + current_node.delays[neighbor]
            new_f_score = g_scores[current_node] + new_distance + h_scores[neighbor]

            # If the new distance and delay sum are shorter than the current distance and delay sum to the neighbor
            if new_distance < distances[neighbor] and new_delay_sum < delay_threshold:
                # Update the distance, previous node, delay sum, and f_score
                distances[neighbor] = new_distance
                previous[neighbor] = current_node
                delay_sums[neighbor] = new_delay_sum
                f_scores[neighbor] = new_f_score

                # If the neighbor is not in the open set, add it
                if neighbor not in open_set:
                    open_set.append(neighbor)

    # Reconstruct the path from the destination to the source
    path = []
    current_node = destination
    while current_node is not None:
        path.append(current_node)
        current_node = previous[current_node]

    # Reverse the path to get the path from the source to the destination
    path.reverse()

    # Check if the path satisfies the bandwidth demand
    if not check_bandwidth_demand(path, bandwidth_demand):
        return None

    return path

def dijkstra_with_reliability_constraint(nodes, edges, source, destination, bandwidth_demand, reliability_threshold):
    # Initialize distances, previous nodes, and reliability products
    distances = {node: float('inf') for node in nodes}
    distances[source] = 0
    previous = {node: None for node in nodes}
    reliability_products = {node: 1.0 for node in nodes}

    # Initialize priority queue
    pq = [(0, 1.0, source)]

    # While there are still nodes to visit
    while pq:
        # Get the node with the smallest distance and reliability product
        current_distance, current_reliability_product, current_node = heapq.heappop(pq)

        # If we have reached the destination, we are done
        if current_node == destination:
            break

        # For each neighbor of the current node
        for neighbor in current_node.neighbors:
            # Calculate the new distance and reliability product to the neighbor
            new_distance = current_distance + current_node.distances[neighbor]
            new_reliability_product = current_reliability_product * current_node.reliabilities[neighbor]

            # If the new distance and reliability product are shorter than the current distance and reliability product to the neighbor
            if new_distance < distances[neighbor] and new_reliability_product > reliability_threshold:
                # Update the distance, previous node, reliability product, and f_score
                distances[neighbor] = new_distance
                previous[neighbor] = current_node
                reliability_products[neighbor] = new_reliability_product

                # Add the neighbor to the priority queue
                heapq.heappush(pq, (new_distance, new_reliability_product, neighbor))

    # Reconstruct the path from the destination to the source
    path = []
    current_node = destination
    while current_node is not None:
        path.append(current_node)
        current_node = previous[current_node]

    # Reverse the path to get the path from the source to the destination
    path.reverse()

    # Check if the path satisfies the bandwidth demand
    if not check_bandwidth_demand(path, bandwidth_demand):
        return None

    return path

def bellman_ford_with_reliability_constraint(nodes, edges, source, destination, bandwidth_demand, reliability_threshold):
    # Initialize distances, previous nodes, and reliability products
    distances = {node: float('inf') for node in nodes}
    distances[source] = 0
    previous = {node: None for node in nodes}
    reliability_products = {node: 1.0 for node in nodes}

    # Relax all edges |V| - 1 times
    for _ in range(len(nodes) - 1):
        for edge in edges:
            # If the new distance and reliability product to the neighbor are shorter than the current distance and reliability product to the neighbor
            if distances[edge.node1] + edge.distance < distances[edge.node2] and reliability_products[edge.node1] * edge.reliability > reliability_threshold:
                # Update the distance, previous node, reliability product, and f_score
                distances[edge.node2] = distances[edge.node1] + edge.distance
                previous[edge.node2] = edge.node1
                reliability_products[edge.node2] = reliability_products[edge.node1] * edge.reliability

    # Check for negative weight cycles
    for edge in edges:
        if distances[edge.node1] + edge.distance < distances[edge.node2] and reliability_products[edge.node1] * edge.reliability > reliability_threshold:
            # Negative weight cycle found, return None
            return None

    # Reconstruct the path from the destination to the source
    path = []
    current_node = destination
    while current_node is not None:
        path.append(current_node)
        current_node = previous[current_node]

    # Reverse the path to get the path from the source to the destination
    path.reverse()

    # Check if the path satisfies the bandwidth demand
    if not check_bandwidth_demand(path, bandwidth_demand):
        return None

    return path

def a_star_with_reliability_constraint(nodes, edges, source, destination, bandwidth_demand, reliability_threshold):
    # Initialize open and closed sets
    open_set = [source]
    closed_set = set()

    # Initialize g_scores, h_scores, and reliability products
    g_scores = {node: float('inf') for node in nodes}
    g_scores[source] = 0
    h_scores = {node: math.inf for node in nodes}
    h_scores[destination] = 0
    reliability_products = {node: 1.0 for node in nodes}

    # Initialize f_scores
    f_scores = {node: float('inf') for node in nodes}
    f_scores[source] = h_scores[source]

    # While the open set is not empty
    while open_set:
        # Get the node with the lowest f_score
        current_node = min(open_set, key=lambda node: f_scores[node])

        # If we have reached the destination, we are done
        if current_node == destination:
            break

        # Move the current node from the open set to the closed set
        open_set.remove(current_node)
        closed_set.add(current_node)

        # For each neighbor of the current node
        for neighbor in current_node.neighbors:
            # If the neighbor is in the closed set, skip it
            if neighbor in closed_set:
                continue

            # Calculate the new distance, reliability product, and f_score
            new_distance = current_node.distances[neighbor]
            new_reliability_product = reliability_products[current_node] * current_node.reliabilities[neighbor]
            new_f_score = g_scores[current_node] + new_distance + h_scores[neighbor]

            # If the new distance





            # If the new distance and reliability product are shorter than the current distance and reliability product to the neighbor
            if new_distance < distances[neighbor] and new_reliability_product > reliability_threshold:
                # Update the distance, previous node, reliability product, and f_score
                distances[neighbor] = new_distance
                previous[neighbor] = current_node
                reliability_products[neighbor] = new_reliability_product
                f_scores[neighbor] = new_f_score

                # If the neighbor is not in the open set, add it
                if neighbor not in open_set:
                    open_set.append(neighbor)

    # Reconstruct the path from the destination to the source
    path = []
    current_node = destination
    while current_node is not None:
        path.append(current_node)
        current_node = previous[current_node]

    # Reverse the path to get the path from the source to the destination
    path.reverse()

    # Check if the path satisfies the bandwidth demand
    if not check_bandwidth_demand(path, bandwidth_demand):
        return None

    return path
       

# Simulated annealing algorithm
def simulated_annealing(nodes, edges, source, destination, bandwidth_demand):
    # Initialize the current solution and temperature
    current_solution = generate_random_solution(nodes, edges, source, destination)
    temperature = 1000

    # While the temperature is greater than the cooling threshold
    while temperature > 0.01:
        # Generate a random neighbor of the current solution
        neighbor = generate_neighbor(current_solution)

        # Calculate the difference in objective function values between the current solution and the neighbor
        delta_objective = calculate_objective_function(neighbor) - calculate_objective_function(current_solution)

        # If the neighbor is better than the current solution or if the neighbor is worse but still accepted with a certain probability, update the current solution
        if delta_objective < 0 or np.random.rand() < math.exp(-delta_objective / temperature):
            current_solution = neighbor

        # Cool down the temperature
        temperature *= 0.99

    # Return the current solution
    return current_solution

# Tabu search algorithm
def tabu_search(nodes, edges, source, destination, bandwidth_demand):
    # Initialize the current solution, tabu list, and best solution
    current_solution = generate_random_solution(nodes, edges, source, destination)
    tabu_list = []
    best_solution = current_solution

    # Set the tabu list size
    tabu_list_size = 10

    # Set the maximum number of iterations
    max_iterations = 100

    # Initialize the iteration counter
    iteration = 0

    # While the maximum number of iterations has not been reached
    while iteration < max_iterations:
        # Generate a list of all possible neighbors of the current solution
        neighbors = []
        for i in range(len(current_solution) - 1):
            for j in range(i + 1, len(current_solution)):
                neighbor = current_solution.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)

        # Remove any neighbors that are in the tabu list
        neighbors = [neighbor for neighbor in neighbors if neighbor not in tabu_list]

        # If there are no neighbors that are not in the tabu list, stop the search
        if not neighbors:
            break

        # Select the best neighbor that is not in the tabu list
        best_neighbor = max(neighbors, key=calculate_objective_function)

        # Update the current solution and the best solution
        current_solution = best_neighbor
        if calculate_objective_function(current_solution) > calculate_objective_function(best_solution):
            best_solution = current_solution

        # Add the current solution to the tabu list
        tabu_list.append(current_solution)

        # Remove the oldest solution from the tabu list
        if len(tabu_list) > tabu_list_size:
            tabu_list.pop(0)

        # Increment the iteration counter
        iteration += 1

    # Return the best solution
    return best_solution

# Ant colony optimization algorithm
def ant_colony_optimization(nodes, edges, source, destination, bandwidth_demand):
    # Initialize the pheromone matrix
    pheromone_matrix = np.ones((len(nodes), len(nodes)))

    # Initialize the ant colony
    ants = []
    for i in range(100):
        ant = Ant(source, destination)
        ants.append(ant)

    # Set the evaporation rate
    evaporation_rate = 0.5

    # Set the maximum number of iterations
    max_iterations = 100

    # Initialize the iteration counter
    iteration = 0

    # While the maximum number of iterations has not been reached
    while iteration < max_iterations:
        # Let each ant construct a path from the source to the destination
        for ant in ants:
            ant.construct_path(pheromone_matrix)

        # Update the pheromone matrix
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                pheromone_matrix[i][j] *= evaporation_rate
                for ant in ants:
                    if ant.path[i] == j:
                        pheromone_matrix[i][j] += ant.pheromone_level

        # Find the best path found by any ant
        best_path = max(ants, key=lambda ant: ant.path_length)

        # If the best path satisfies the bandwidth demand, return it
        if check_bandwidth_demand(best_path.path, bandwidth_demand):
            return best_path.path

        # Increment the iteration counter
        iteration += 1

    # Return the best path found by any ant
    return best_path.path

# Bee colony optimization algorithm
def bee_colony_optimization(nodes, edges, source, destination, bandwidth_demand):
    # Initialize the food source matrix
    food_source_matrix = np.ones((len(nodes), len(nodes)))

    # Initialize the bee colony
    bees = []
    for i in range(100):
        bee = Bee(source, destination)
        bees.append(bee)

    # Set the evaporation rate
    evaporation_rate = 0.5

    # Set the maximum number of iterations
    max_iterations = 100

    # Initialize the iteration counter
    iteration = 0

    # While the maximum number of iterations has not been reached
    while iteration < max_iterations:
        # Let each bee search for a food source
        for bee in bees:
            bee.search_for_food_source(food_source_matrix)

        # Update the food source matrix
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                food_source_matrix[i][j] *= evaporation_rate
                for bee in bees:
                    if bee.food_source[i] == j:
                        food_source_matrix[i][j] += bee.food_source_level

        # Find the best food source found by any bee
        best_food_source = max(bees, key=lambda bee: bee.food_source_value)

        # If the best food source satisfies the bandwidth demand, return it
        if check_bandwidth_demand(best_food_source.food_source, bandwidth_demand):
            return best_food_source.food_source

        # Increment the iteration counter
        iteration += 1

    # Return the best food source found by any bee
    return best_food_source.food_source

# Firefly algorithm
def firefly_algorithm(nodes, edges, source, destination, bandwidth_demand):
    # Initialize the firefly population
    fireflies = []
    for i in range(100):
        firefly = Firefly(source, destination)
        fireflies.append(firefly)

    # Set the absorption coefficient
    absorption_coefficient = 0.9

    # Set the maximum number of iterations
    max_iterations = 100

    # Initialize the iteration counter
    iteration = 0

    # While the maximum number of iterations has not been reached
    while iteration < max_iterations:
        # Let each firefly search for a better solution
        for firefly in fireflies:
            firefly.search_for_better_solution(fireflies, absorption_coefficient)

        # Find the best solution found by any firefly
        best_firefly = max(fireflies, key=lambda firefly: firefly.objective_function_value)

# Find the best solution found by any firefly


        # If the best solution satisfies the bandwidth demand, return it
        if check_bandwidth_demand(best_firefly.solution, bandwidth_demand):
            return best_firefly.solution

        # Increment the iteration counter
        iteration += 1

    # Return the best solution found by any firefly
    return best_firefly.solution

# Class for representing an ant
class Ant:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
        self.path = []
        self.path_length = 0
        self.pheromone_level = 1

    def construct_path(self, pheromone_matrix):
        # Start at the source node
        current_node = self.source

        # While we have not reached the destination node
        while current_node != self.destination:
            # Select the next node to visit based on the pheromone levels
            next_node = np.random.choice(nodes, p=pheromone_matrix[current_node] / np.sum(pheromone_matrix[current_node]))

            # Add the next node to the path
            self.path.append(next_node)

            # Update the path length
            self.path_length += nodes[current_node].distances[next_node]

            # Update the current node
            current_node = next_node

# Class for representing a bee
class Bee:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
        self.food_source = []
        self.food_source_value = 0
        self.food_source_level = 1

    def search_for_food_source(self, food_source_matrix):
        # Start at the source node
        current_node = self.source

        # While we have not reached the destination node
        while current_node != self.destination:
            # Select the next node to visit based on the food source levels
            next_node = np.random.choice(nodes, p=food_source_matrix[current_node] / np.sum(food_source_matrix[current_node]))

            # Add the next node to the food source
            self.food_source.append(next_node)

            # Update the food source value
            self.food_source_value += nodes[current_node].distances[next_node]

            # Update the current node
            current_node = next_node

# Class for representing a firefly
class Firefly:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
        self.solution = []
        self.objective_function_value = float('inf')

    def search_for_better_solution(self, other_fireflies, absorption_coefficient):
        # Generate a random solution
        new_solution = generate_random_solution(nodes, edges, source, destination)

        # Calculate the objective function value of the new solution
        new_objective_function_value = calculate_objective_function(new_solution)

        # If the new solution is better than the current solution
        if new_objective_function_value < self.objective_function_value:
            # Update the current solution
            self.solution = new_solution
            self.objective_function_value = new_objective_function_value

        # For each other firefly
        for other_firefly in other_fireflies:
            # Calculate the distance between the current firefly and the other firefly
            distance = np.linalg.norm(np.array(self.solution) - np.array(other_firefly.solution))

            # If the other firefly is brighter than the current firefly
            if other_firefly.objective_function_value < self.objective_function_value:
                # Move the current firefly towards the other firefly
                self.solution = self.solution + absorption_coefficient * (other_firefly.solution - self.solution)

                # Update the objective function value of the current solution
                self.objective_function_value = calculate_objective_function(self.solution)

# Function to read the input file
def read_input_file(filename):
    nodes = []
    edges = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Read adjacency matrix
    adjacency_matrix = []
    for line in lines:
        if line.strip():
            adjacency_matrix.append([int(x) for x in line.split(':')])

    # Read bandwidth matrix
    bandwidth_matrix = []
    for line in lines:
        if line.strip():
            bandwidth_matrix.append([int(x) for x in line.split(':')])

    # Read delay matrix
    delay_matrix = []
    for line in lines:
        if line.strip():
            delay_matrix.append([int(x) for x in line.split(':')])

    # Read reliability matrix
    reliability_matrix = []
    for line in lines:
        if line.strip():
            reliability_matrix.append([float(x) for x in line.split(':')])

    # Create nodes
    for i in range(len(adjacency_matrix)):
        node = Node(i)
        nodes.append(node)

    # Create edges
    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix)):
            if adjacency_matrix[i][j] == 1:
                distance = random.randint(1, 5)
                bandwidth = random.randint(3, 10)
                delay = random.randint(1, 5)
                reliability = random.uniform(0.95, 0.99)
                edge = Edge(nodes[i], nodes[j], distance, bandwidth, delay, reliability)
                edges.append(edge)
                nodes[i].neighbors.append(nodes[j])
                nodes[i].distances[nodes[j]] = distance
                nodes[i].bandwidths[nodes[j]] = bandwidth
                nodes[i].delays[nodes[j]] = delay
                nodes[i].reliabilities[nodes[j]] = reliability

    return nodes, edges

# Function to check if the bandwidth demand is satisfied
def check_bandwidth_demand(path, bandwidth_demand):
    # Check if the bandwidth of each edge in the path is greater than or equal to the bandwidth demand
    for i in range(len(path) - 1):
        if path[i].bandwidths[path[i + 1]] < bandwidth_demand:
            return False

    return True

# Function to calculate the objective function value of a solution for metaheuristic algorithms
def calculate_objective_function(solution):
    # Calculate the total distance of the path
    total_distance = 0
    for i in range(len(solution) - 1):
        total_distance += solution[i].distances[solution[i + 1]]

    # Calculate the delay of the path
    total_delay = 0
    for i in range(len(solution) - 1):
        total_delay += solution[i].delays[solution[i + 1]]

    # Calculate the reliability of the path
    total_reliability = 1
    for i in range(len(solution) - 1):
        total_reliability *= solution[i].reliabilities[solution[i + 1]]

    # Calculate the objective function value
    objective_function_value = 0.5 * total_distance + 0.25 * total_delay + 0.25 * total_reliability

    return objective_function_value

# Main function
if __name__ == "__main__":
    # Read the input file and initialize the graph
    nodes, edges = read_input_file("USNET_AjdMatrix.txt")

    print(f"Number of nodes read: {len(nodes)}")

    # Define source and destination nodes
    # Ensure these are valid based on your network
    source_id = 0
    destination_id = 24

    # Validate source and destination
    if source_id < 0 or source_id >= len(nodes) or destination_id < 0 or destination_id >= len(nodes):
        print("Invalid source or destination ID.")
        exit()

    # Set the bandwidth demand
    bandwidth_demand = 5

    # Call each algorithm and print results
    
for algorithm in [dijkstra, bellman_ford, a_star, dijkstra_with_delay_constraint, bellman_ford_with_delay_constraint, a_star_with_delay_constraint, dijkstra_with_reliability_constraint, bellman_ford_with_reliability_constraint, a_star_with_reliability_constraint, simulated_annealing, tabu_search, ant_colony_optimization, bee_colony_optimization, firefly_algorithm]:
        path = algorithm(nodes, edges, source_id, destination_id, bandwidth_demand)

        # Check if the path is valid
        if path is None:
            print(f"{algorithm.__name__}: No path found.")
        else:
            # Print the path and its objective function value
            print(f"{algorithm.__name__}:")
            print(f"Path: {path}")
            print(f"Objective function value: {calculate_objective_function(path)}")

            # Check if the path satisfies the bandwidth demand
            if not check_bandwidth_demand(path, bandwidth_demand):
                print(f"{algorithm.__name__}: Path does not satisfy the bandwidth demand.")