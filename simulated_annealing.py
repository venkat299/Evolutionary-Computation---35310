# import enum
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
                        persistence=1, 
                        debug=True):

    
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
        if debug:
            print(f'iteration : {i}, current: {current_cost}, neighbour: {neighbor_cost},  temp: {round(temperature,3)}, explored:{len(explored)}, feasible:{len(explored)-len(infeasible)}, manhattan:{similarity}')
    
    return best_solution, best_cost



def generate_neighbor(solution, problem, explored, infeasible, persistence=1):
    
    def swap(solution, swap_size=1):
        neighbor = solution.copy() 
        ones = [i for i, x in enumerate(neighbor) if x == 1]
        if ones:
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
        
        I = problem["rows"].keys() # Set of rows 
        U = set(I)  # Initialize U as the set of all rows
        Sk = set()  # Initialize an empty solution
        for idx, val in enumerate(neighbor): 
            if val==1:
                Sk.add(idx)
                for row in problem["columns"][idx]:
                    U.remove(row)
        
        # print(U, swap_size)
        while U:
            # Randomly select a row i from U
            i = random.choice(list(U))
            
            # Randomly select a column j from rows[i] such that β[j] ∩ (I − U) = ∅
            valid_columns = [j for j in problem["rows"][i] if not problem["columns"][j].intersection(set(I) - U)]
            
            if valid_columns:
                # Add column j to Sk, and update U
                j = random.choice(valid_columns)
                Sk.add(j)
                U -= problem["columns"][j]  # Remove all rows covered by column j from U
            else:
                # If no valid column exists, remove i from U
                U.remove(i)

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
    covered_rows = set()  
    row_counts = {}  
    
    
    for i, selected in enumerate(solution):
        if selected:  
            covered_rows.update(problem['columns'][i])
            for row in problem['columns'][i]:
                if row in row_counts:
                    row_counts[row] += 1
                else:
                    row_counts[row] = 1

    for row in problem['rows'].keys():
        if row not in row_counts or row_counts[row] != 1:
            return False  
    return len(covered_rows) == len(problem['rows'])  

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


def initialize_population(population_size, problem_data):
    population_size = max(1, population_size)
    rows = problem_data["rows"]
    columns = problem_data["columns"]
    population = []
    row_indices = rows.keys()
    count = 0
    while count < population_size:
        solution_set = set()
        uncovered_rows = set(row_indices)
        while uncovered_rows:
            row_index = random.choice(list(uncovered_rows))
            valid_columns = [column_index for column_index in rows[row_index] if not columns[column_index].intersection(set(row_indices) - uncovered_rows)]
            if valid_columns:
                column_index = random.choice(valid_columns)
                solution_set.add(column_index)
                uncovered_rows -= columns[column_index]
            else:
                uncovered_rows.remove(row_index)
        solution = [0] * len(problem_data["columns"])
        for index in solution_set:
            solution[index] = 1
        if is_feasible(solution, problem_data):
            population.append(solution)
            count = count + 1
        else:
            continue
    return population




def invert_dict_of_sets(d):
    inverted = {}
    for key, value_set in d.items():
        for v in value_set:
            if v not in inverted:
                inverted[v] = set()
            inverted[v].add(key)
    return inverted



def manhattan_similarity(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")
    
    # Calculate Manhattan distance
    distance = sum(abs(a - b) for a, b in zip(list1, list2))
    
    return distance


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


def sa(file_path, debug=False):
    rows, columns, costs = read_set_cover_data(file_path)
    problem = {}
    problem["rows"] = rows  
    problem["columns"] = columns
    problem["cost"]= costs
    
    solution = initialize_population(1, problem)[0]

    simulated_annealing_params = {
                        "problem":problem, 
                        "calculate_cost":calculate_cost, 
                        "get_neighbor":generate_neighbor, 
                        "initial_solution":solution, 
                        "initial_temp":100, 
                        "cooling_rate":0.995, 
                        "iterations":1000,
                        "persistence":1,
                        "debug":False
                        # "min_temp": 0.1,
    }
    
    best_solution, best_cost = simulated_annealing(**simulated_annealing_params)
    selected_columns = [idx for idx, val in enumerate(best_solution) if val==1]
    # # print("best_solution", best_solution)
    # print("selected_columns", selected_columns)
    # print("cost of solution", best_cost)
    
    check_list = []
    for i in selected_columns:
        # print(list(columns[i]))
        check_list.extend(list(columns[i]))
    # check_list.sort()
    # print(check_list)
    
    # print("row count", len(rows))
    # print("length", len(check_list))
    # print("length", len(set(check_list)))
    # print("feasible: ",is_feasible(best_solution, problem))
    feasible = is_feasible(best_solution, problem)
    
    return best_solution, best_cost, feasible
    


if __name__ == "__main__":

    file_path = './sppnw42.txt'  
    rows, columns, costs = read_set_cover_data(file_path)
    
    # Print the results
    print("num_rows:", len(rows))
    print("num_columns:", len(columns))
    # print("costs:", costs)
    # print("rows:", rows)
    # print("columns:", columns)
    
    problem = {}
    problem["rows"] = rows  
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


