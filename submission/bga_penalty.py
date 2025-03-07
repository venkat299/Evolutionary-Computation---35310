import numpy as np
import random
import matplotlib.pyplot as plt



def simple_ga_set_partition(matrix_a, column_cost, max_iter=100, pop_size=200, penalty=10000):
    """
    Genetic Algorithm for Set Partitioning (Air Crew Scheduling)
    
    Args:
        matrix_a (ndarray): Constraint matrix (num_constraints x num_columns)
        column_cost (ndarray): Cost vector (num_columns,)
        max_iter (int): Maximum iterations
        pop_size (int): Population size
    
    Returns:
        best_solution (ndarray): Best individual found
        best_cost (float): Minimum cost found
    """
    num_columns = matrix_a.shape[1]
    
    # Initialize population randomly (0 or 1)
    population = np.random.rand(pop_size, num_columns) > 0.5
    # population = initialize_population()

    # Track best and average fitness values
    best_fitness_list = []
    avg_fitness_list = []

    for iteration in range(max_iter):
        # Evaluate fitness
        fitness, total_cost, total_violations = cal_fitness(population, matrix_a, column_cost, penalty)

        # Sort population by fitness (ascending, since we minimize cost)
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]

        # Store statistics
        best_fitness_list.append(fitness[0])  # Best fitness (lowest cost)
        avg_fitness_list.append(np.mean(fitness))  # Average fitness
        
        # Selection: Top 30% parents
        num_parents = int(0.3 * pop_size)
        # population[:num_parents]
        parents = tournament_selection(population, fitness, tournament_size=3)
        
        # Uniform crossover
        offspring = uniform_crossover(parents, pop_size, num_columns)

        # Mutation: Flip a random bit in each offspring
        mutation_prob = random.uniform(1/num_columns, 0.8)# Mutation probability
        for j in range(num_parents):
            if np.random.rand() < mutation_prob:  # Apply mutation with probability
                mutation_bit = np.random.randint(num_columns)
                offspring[j, mutation_bit] = 1 - offspring[j, mutation_bit]  # Flip bit
        

        
        # Merge offspring into population
        population = np.vstack((population, offspring))
        

        # Recalculate fitness and keep best PopSize individuals
        fitness, total_cost, total_violations = cal_fitness(population, matrix_a, column_cost, penalty)
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices[:pop_size]]
        
        print(f'generation:{iteration}, cost:{total_cost[0]}, violation:{total_violations[0]}, variance:{round(np.std(total_cost))}')


        # Stopping criteria
        if iteration == max_iter - 1:
            print(f"Final iteration reached: {max_iter}")
    
    # Get the best solution
    best_solution = population[0]
    best_cost = total_cost[0]
    best_violations = total_violations[0]

    # Print results
    print(f"Minimum cost found by GA: {best_cost}")
    print(f"Total constraint violations: {best_violations}")

    # # Plot fitness trends
    # plt.plot(best_fitness_list, label="Best Fitness (Cost)")
    # plt.plot(avg_fitness_list, label="Average Fitness (Cost)", linestyle="--")
    # plt.xlabel("Iteration")
    # plt.ylabel("Cost")
    # plt.legend()
    # plt.title("GA Convergence")
    # plt.show()

    return best_solution, best_cost

def uniform_crossover(parents, pop_size, num_bits):
    """
    Perform Uniform Crossover on the parent population.

    Args:
        parents (ndarray): Parent population matrix (PopSize x num_bits).
        pop_size (int): Population size.
        num_bits (int): Number of bits (columns).
        
    Returns:
        ndarray: Offspring population matrix (PopSize x num_bits).
    """
    offspring = np.zeros_like(parents)

    for i in range(0, pop_size - 1, 2):  # Pair up parents in the population
        # For each pair of parents, perform uniform crossover
        parent1 = parents[i]
        parent2 = parents[i + 1]

        # For each bit in the individual, randomly pick from either parent
        crossover_mask = np.random.randint(0, 2, num_bits)  # Random mask (0 or 1 for each gene)
        offspring[i] = np.where(crossover_mask == 0, parent1, parent2)  # If mask is 0, pick from parent1; else from parent2
        offspring[i + 1] = np.where(crossover_mask == 0, parent2, parent1)  # Swap parents for the next individual

    return offspring


def tournament_selection(population, fitness, tournament_size=3):
    """
    Perform tournament selection on a population.

    Args:
        population (ndarray): Population matrix (PopSize x num_bits).
        fitness (ndarray): Array of fitness values for each individual.
        tournament_size (int): Number of individuals to compete in a tournament.

    Returns:
        ndarray: Selected parents (same size as original population).
    """
    pop_size = population.shape[0]
    num_bits = population.shape[1]
    
    selected_parents = np.zeros((pop_size, num_bits), dtype=int)  # Placeholder for selected individuals

    for i in range(pop_size):
        # Select 'tournament_size' random individuals
        tournament_indices = random.sample(range(pop_size), tournament_size)
        tournament_fitness = fitness[tournament_indices]
        
        # Find the best individual in the tournament
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]  # Minimizing cost (lower fitness is better)
        
        # Add the winner to the selected parents
        selected_parents[i] = population[winner_idx]

    return selected_parents


def cal_fitness(pop, matrix_a, column_cost, penalty):
    """
    Calculate fitness of each individual in the population.
    
    Fitness = total_cost + (penalty * constraint violations)
    
    Args:
        pop (ndarray): Population matrix (PopSize x num_columns)
        matrix_a (ndarray): Constraint matrix (num_constraints x num_columns)
        column_cost (ndarray): Cost vector (num_columns,)
    
    Returns:
        fitness (ndarray): Fitness values for each individual
        total_cost (ndarray): Cost values for each individual
        total_violations (ndarray): Constraint violations for each individual
    """
    pop_size = pop.shape[0]
    
    # Calculate total cost (sum of selected column costs)
    total_cost = np.sum(pop * column_cost, axis=1)
    
    # Calculate constraint violations
    constraint_violation = np.sum((np.dot(matrix_a, pop.T) - 1) ** 2, axis=0)
    
    # Fitness function (minimize cost + penalty for violations)
    # penalty = 50000  # High penalty for constraint violations
    fitness = total_cost + penalty * constraint_violation
    
    return fitness, total_cost, constraint_violation


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
    population_size = max(1, population_size)
    rows = problem["rows"]
    columns = problem["columns"]
    population = []
    I = rows.keys() # Set of rows (flights to cover).
    k=0
    while k<population_size:
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
        if is_feasible(solution, problem):
            population.append(solution)
            k=k+1
        else:
            continue

    
    return population


def convert_to_dicts(matrix_a):
    """
    Converts the constraint matrix into dictionaries:
    - `rows`: Maps each row to the set of columns covering it.
    - `columns`: Maps each column to the set of rows it covers.

    Args:
        matrix_a (ndarray): Constraint matrix (num_constraints x num_columns).

    Returns:
        rows (dict): {row_index: set(columns)}
        columns (dict): {column_index: set(rows)}
    """
    num_constraints, num_columns = matrix_a.shape
    rows = {i: set() for i in range(num_constraints)}
    columns = {j: set() for j in range(num_columns)}

    for i in range(num_constraints):
        for j in range(num_columns):
            if matrix_a[i, j] == 1:  # If column j covers row i
                rows[i].add(j)
                columns[j].add(i)

    return rows, columns


def load_dataset(filename):
    """
    Reads the set partitioning problem dataset from a file.

    Args:
        filename (str): Path to the dataset file.

    Returns:
        matrix_a (ndarray): Constraint matrix (num_constraints x num_columns)
        column_cost (ndarray): Cost vector (num_columns,)
    """
    with open(filename, 'r') as file:
        # Read the first line: number of rows (constraints) and number of columns
        num_constraints, num_columns = map(int, file.readline().split())

        # Initialize constraint matrix and cost vector
        matrix_a = np.zeros((num_constraints, num_columns), dtype=int)
        column_cost = np.zeros(num_columns, dtype=int)

        # Read each column data
        for j in range(num_columns):
            data = list(map(int, file.readline().split()))
            column_cost[j] = data[0]  # First value is the column cost
            covered_rows = data[2:]  # Remaining values are the covered rows
            for row in covered_rows:
                matrix_a[row - 1, j] = 1  # Convert to zero-based index

    return matrix_a, column_cost



# Example Usage
if __name__ == "__main__":
    # # Example Problem Data
    # num_constraints = 5
    # num_columns = 10
    
    # # Random Constraint Matrix (Binary 0/1)
    # matrix_a = np.random.randint(0, 2, (num_constraints, num_columns))

    # # Random Column Costs
    # column_cost = np.random.randint(10, 50, num_columns)
    # Example usage
    filename = "./data/set_cover/sppnw42.txt"  # Change this to your actual file path
    matrix_a, column_cost = load_dataset(filename)
    
    # Example usage
    rows_dict, columns_dict = convert_to_dicts(matrix_a)
    problem = {}
    problem["rows"] = rows_dict
    problem["columns"] = columns_dict
    problem["cost"] = column_cost

    # Print sample data
    print("row count:" ,len(rows_dict))
    print("column count:" ,len(columns_dict))
    # print("Rows (first 5):", {k: v for k, v in list(rows_dict.items())[:5]})
    # print("Columns (first 5):", {k: v for k, v in list(columns_dict.items())[:5]})


    # Run Genetic Algorithm
    best_solution, best_cost = simple_ga_set_partition(matrix_a, column_cost, max_iter=200, pop_size=1000, penalty=100000)
    # print(best_solution, best_cost)
