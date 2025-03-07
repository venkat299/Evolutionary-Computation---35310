import enum
import random
import math


import random
# random.seed(19)

def simulated_annealing(problem, 
                        calculate_cost, 
                        get_neighbor, 
                        initial_solution, 
                        initial_temp, 
                        cooling_rate, 
                        iterations,
                        persistence=1,):
    
    current_solution = initial_solution
    current_cost = calculate_cost(current_solution, problem)
    best_solution = current_solution
    best_cost = current_cost
    temperature = initial_temp
    
    explored = set()
    infeasible = set()
    explored.add(str(best_solution))

    for i in range(iterations):
        neighbor, explored, infeasible = get_neighbor(current_solution, problem, explored, infeasible, persistence)
        # print(neighbor)
        
        neighbor_cost = calculate_cost(neighbor, problem)
        
        if neighbor_cost < current_cost or random.random() < math.exp((current_cost - neighbor_cost) / temperature):
            current_solution = neighbor
            current_cost = neighbor_cost
        
        if current_cost < best_cost:
            best_solution = current_solution
            best_cost = current_cost
        
        temperature *= cooling_rate
        similarity = manhattan_similarity(current_solution, neighbor)
        print(f'iteration : {i}, current: {current_cost}, neighbour: {neighbor_cost},  temp: {round(temperature,3)}, explored:{len(explored)}, feasible:{len(explored)-len(infeasible)}, manhattan:{similarity}')
    
    return best_solution, best_cost



def generate_neighbor(solution, problem, explored, infeasible, persistence=1):
    """
    Generate a neighbor solution by swapping two elements.
    
    Args:
    solution (list): Current solution represented as a binary list.
    problem (dict): Problem instance containing constraints.
    
    Returns:
    list: New neighbor solution after the swap operation.
    """ 
    def swap(solution, swap_size=1):
        neighbor = solution.copy()  # Create a copy of the current solution
    
        # Get indices of 1s and 0s in the solution
        ones = [i for i, x in enumerate(neighbor) if x == 1]
        # zeros = [i for i, x in enumerate(neighbor) if x == 0]
    
        # Ensure there's at least one 1 and one 0 to swap
        if ones:
            # Randomly select one index with 1 and one with 0
            # index_one = random.choices(ones, swap_size)
            swap_size = max(2, swap_size)
            # print("counter", counter, "swap_size", swap_size, "column_count", len(ones) )
            if swap_size>=len(ones):
                # print("getting new candidate", "swap size", swap_size)
                return initialize_population(1, problem)[0]
            
            indices = random.sample(ones, swap_size)
            # index_zero = random.choice(zeros)
        
            # Set the index to zero
            for i in indices:
                neighbor[i] = 0
        
        I = problem["rows"].keys() # Set of rows (flights to cover).
        U = set(I)  # Initialize U as the set of all rows
        Sk = set()  # Initialize an empty solution
        for idx, val in enumerate(neighbor): 
            if val==1:
                Sk.add(idx)
                for row in columns[idx]:
                    U.remove(row)
        
        # print(U, swap_size)
        while U:
            # Randomly select a row i from U
            i = random.choice(list(U))
            
            # Randomly select a column j from rows[i] such that β[j] ∩ (I − U) = ∅
            valid_columns = [j for j in rows[i] if not columns[j].intersection(set(I) - U)]
            
            if valid_columns:
                # Add column j to Sk, and update U
                j = random.choice(valid_columns)
                Sk.add(j)
                U -= columns[j]  # Remove all rows covered by column j from U
            else:
                # If no valid column exists, remove i from U
                U.remove(i)

        # Add the generated solution Sk to the population
        solution = [0]*len(problem["columns"])
        for i in Sk:
            solution[i]=1
        # print(Sk, U, )
        
        return solution
    
    counter = 0
    column_count = len([i for i, x in enumerate(solution) if x == 1])
    
    while persistence>0:
        persistence = persistence-1
        counter = counter + 1
        swap_size = (round(counter%column_count))
        swap_size = min(swap_size, column_count)
        neighbor = swap(solution, swap_size)
        if str(neighbor) in explored:
            continue
        # Check if the new solution is feasible
        if is_feasible(neighbor, problem):
            explored.add(str(neighbor))
            return neighbor, explored, infeasible
        else:
            explored.add(str(neighbor))
            infeasible.add(str(neighbor))
            continue
    # If swap didn't produce current_costa feasible solution, return the original
    return solution, explored, infeasible

def is_feasible(solution, problem):
    """
    Check if the solution is feasible for the Set Partitioning Problem (SPP).
    
    Args:
    solution (list): Solution to check (binary list where 1 indicates a selected column).
    problem (dict): Problem instance containing constraints:
        - problem['columns']: A list of sets, where each set represents the rows covered by a column.
        - problem['rows']: A set of all rows that need to be covered.
    
    Returns:
    bool: True if the solution is feasible, False otherwise.
    """
    covered_rows = set()  # To track rows that are covered
    row_counts = {}  # To track how many times each row is covered

    # Iterate over selected columns in the solution
    for i, selected in enumerate(solution):
        if selected:  # If column i is selected
            covered_rows.update(problem['columns'][i])
            for row in problem['columns'][i]:
                if row in row_counts:
                    # If a row is already covered, increment its count
                    row_counts[row] += 1
                else:
                    # Otherwise, initialize its count to 1
                    row_counts[row] = 1

    # Check completeness: All rows must be covered exactly once
    for row in problem['rows'].keys():
        if row not in row_counts or row_counts[row] != 1:
            return False  # Either not covered or covered more than once

    return len(covered_rows) == len(problem['rows'])  # Feasible if all rows are covered exactly once


# calculate cost
def calculate_cost(solution, problem):
    costs = problem["cost"]
    return sum([costs[idx] for idx, col in  enumerate(solution) if col==1])

def pseudo_random_init(rows, columns):
    S = [0] * len(columns)
    U = set(rows)
    while U:
        j = random.choice(list(U))
        I = [i for i, pairing in enumerate(columns) if j in pairing]
        if I:
            k = random.choice(I)
            S[k] = 1
            U -= set(columns[k])
    return S


def initialize_population(population_size, problem):
    """
    Initialize the population P(0) for the Set Partitioning Problem (SPP).
    
    Parameters:
    - population_size : Number of individuals in the population.
    - problem->rows: A dictionary where rows[i] is the set of columns (pairings) that cover row i.
    - problem->columns: A dictionary where columns[j] is the set of rows covered by column j.
    
    Returns:
    - population: A list of solutions, where each solution is a set of selected columns.
    """
    rows = problem["rows"]
    columns = problem["columns"]
    population = []
    I = rows.keys() # Set of rows (flights to cover).
    for k in range(population_size):
        Sk = set()  # Initialize an empty solution
        U = set(I)  # Initialize U as the set of all rows

        while U:
            # Step 4: Randomly select a row i from U
            i = random.choice(list(U))
            
            # Step 5: Randomly select a column j from rows[i] such that β[j] ∩ (I − U) = ∅
            valid_columns = [j for j in rows[i] if not columns[j].intersection(set(I) - U)]
            
            if valid_columns:
                # Step 7: Add column j to Sk, and update U
                j = random.choice(valid_columns)
                Sk.add(j)
                U -= columns[j]  # Remove all rows covered by column j from U
            else:
                # Step 9: If no valid column exists, remove i from U
                U.remove(i)

        # Add the generated solution Sk to the population
        solution = [0]*len(problem["columns"])
        for i in Sk:
            solution[i]=1
        population.append(solution)

    
    return population




def invert_dict_of_sets(d):
    """
    Inverts a dictionary of sets such that keys become values and values become keys.
    
    Parameters:
    - d: A dictionary where values are sets.
    
    Returns:
    - inverted: The inverted dictionary.
    """
    inverted = {}
    for key, value_set in d.items():
        for v in value_set:
            if v not in inverted:
                inverted[v] = set()
            inverted[v].add(key)
    return inverted



def manhattan_similarity(list1, list2):
    """
    Measure the similarity between two binary lists using Manhattan distance.

    Args:
    - list1 (list): First binary list.
    - list2 (list): Second binary list.

    Returns:
    - int: Manhattan distance (number of differing positions).
    """
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")
    
    # Calculate Manhattan distance
    distance = sum(abs(a - b) for a, b in zip(list1, list2))
    
    return distance

# Example Usage
binary_list1 = [1, 0, 1, 1, 0]
binary_list2 = [0, 0, 1, 0, 1]

distance = manhattan_similarity(binary_list1, binary_list2)
print(f"Manhattan Distance: {distance}")


def read_set_cover_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # First row contains num_rows and num_columns
    num_rows, num_columns = map(int, lines[0].split())
    # Initialize lists to store costs and coverage information
    costs = []
    coverage = []
    # Process each subsequent line for costs and coverage
    for line in lines[1:]:
        # Split the line into integers
        data = list(map(int, line.split()))
        # The first number is the cost of the set
        cost = data[0]
        costs.append(cost)
        # The remaining numbers are the elements covered by this set
        elements = data[2:]
        if len(elements) != data[1]:
            raise Exception("set length not matching")
        coverage.append([i-1 for i in elements])
    columns = dict(enumerate(coverage))
    for item in columns:
        columns[item] = set(columns[item])
    rows = invert_dict_of_sets(columns)
    return  rows, columns, costs



if __name__ == "__main__":

    # Example usage:
    file_path = './data/set_cover/sppnw42.txt'  # Replace with the actual file path
    rows, columns, costs = read_set_cover_data(file_path)
    
    # costs = [300, 400, 500,600, 700]

    # rows = {
    # 0: {4, 0},
    # 1: {1, 2},
    # 2: {2, 3},print("constraint satisfied :", is_feasible(best_solution, problem_data))
    # 3: {3, 4}
    #   }  # Columns covering each row
    # columns = {
    # 0: {0},
    # 1: {1},
    # 2: {1, 2},
    # 3: {2, 3},
    # 4: {3, 0},
    
    # }  # Rows covered by each column
    
    
    # Print the results
    print("num_rows:", len(rows))
    print("num_columns:", len(columns))
    # print("costs:", costs)
    # print("rows:", rows)
    # print("columns:", columns)
    
    problem = {}
    problem["rows"] = rows  # adjusting index start to zero
    problem["columns"] = columns
    problem["cost"]= costs
    
    solution = initialize_population(1, problem)[0]
    
    # print("initial solution",solution, is_feasible(solution, problem))
    # print(initial_sol)
    print(calculate_cost(solution, problem))
    column_count = len(problem["columns"])
    simulated_annealing_params = {
                        "problem":problem, 
                        "calculate_cost":calculate_cost, 
                        "get_neighbor":generate_neighbor, 
                        "initial_solution":solution, 
                        "initial_temp":100, 
                        "cooling_rate":0.995, 
                        "iterations":1000,
                        "persistence":1
                        # "min_temp": 0.1,
    }
    
    # Run Simulated Annealing
    best_solution, best_cost = simulated_annealing(**simulated_annealing_params)
    selected_columns = [idx for idx, val in enumerate(best_solution) if val==1]
    # print("best_solution", best_solution)
    print("selected_columns", selected_columns)
    print("cost of solution", best_cost)
    
    check_list = []
    for i in selected_columns:
        # print(list(columns[i]))
        check_list.extend(list(columns[i]))
    check_list.sort()
    print(check_list)
    
    print("row count", len(rows))
    print("length", len(check_list))
    print("length", len(set(check_list)))
    print("feasible: ",is_feasible(best_solution, problem))


