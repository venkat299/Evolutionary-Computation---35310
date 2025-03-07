import numpy as np
import random

def genetic_algorithm_set_partition(constraint_matrix, column_costs, max_iterations=100, population_size=200, constraint_violation_penalty=10000):
    
    num_columns = constraint_matrix.shape[1]
    # Initialize population randomly (0 or 1)
    population = np.random.rand(population_size, num_columns) > 0.5
    best_fitness_values = []
    average_fitness_values = []

    for iteration in range(max_iterations):
        # Evaluate fitness
        fitness_values, best_fitness, total_constraint_violations = calculate_fitness(population, constraint_matrix, column_costs, constraint_violation_penalty)
        # Sort population by fitness (ascending, since we minimize cost)
        sorted_indices = np.argsort(fitness_values)
        population = population[sorted_indices]
        # Store statistics
        best_fitness_values.append(fitness_values[0])
        average_fitness_values.append(np.mean(fitness_values))
        # Selection: Tournament selection
        parents = perform_tournament_selection(population, fitness_values, tournament_size=3)
        # Crossover: Uniform crossover
        offspring = perform_uniform_crossover(parents, population_size, num_columns)
        # Mutation: Flip bits
        offspring = perform_mutation(offspring, num_columns)
        # Merge offspring into population
        population = np.vstack((population, offspring))
        # Recalculate fitness and keep best population_size individuals
        fitness_values, best_fitness, total_constraint_violations = calculate_fitness(population, constraint_matrix, column_costs, constraint_violation_penalty)
        sorted_indices = np.argsort(fitness_values)
        population = population[sorted_indices[:population_size]]
        # Print progress
        print(f'generation:{iteration}, violation:{total_constraint_violations[0]}, cost min:{best_fitness[0]}, cost max:{max(best_fitness)}, std:{round(np.std(best_fitness))}')
        
        # Stopping criteria
        if iteration == max_iterations - 1:
            print(f"Final iteration reached: {max_iterations}")
    # Get the best solution
    best_solution = population[0]
    best_cost = best_fitness[0]
    best_violations = total_constraint_violations[0]
    # Print results
    print(f"Minimum cost found by GA: {best_cost}")
    print(f"Total constraint violations: {best_violations}")
    return best_solution, best_cost

def perform_uniform_crossover(parents, population_size, num_bits):
    offspring = np.zeros_like(parents)
    for i in range(0, population_size - 1, 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]
        crossover_mask = np.random.randint(0, 2, num_bits)
        offspring[i] = np.where(crossover_mask == 0, parent1, parent2)
        offspring[i + 1] = np.where(crossover_mask == 0, parent2, parent1)
    return offspring

def perform_tournament_selection(population, fitness_values, tournament_size=3):
    population_size = population.shape[0]
    num_bits = population.shape[1]
    selected_parents = np.zeros((population_size, num_bits), dtype=int)
    for i in range(population_size):
        tournament_indices = random.sample(range(population_size), tournament_size)
        tournament_fitness = fitness_values[tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        selected_parents[i] = population[winner_idx]
    return selected_parents

def calculate_fitness(population, constraint_matrix, column_costs, constraint_violation_penalty):
    population_size = population.shape[0]
    best_fitness = np.sum(population * column_costs, axis=1)
    constraint_violations = np.sum((np.dot(constraint_matrix, population.T) - 1) ** 2, axis=0)
    fitness_values = best_fitness + constraint_violation_penalty * constraint_violations
    return fitness_values, best_fitness, constraint_violations

def perform_mutation(offspring, num_columns):
    mutation_probability = random.uniform(1/num_columns, 0.8)
    for j in range(len(offspring)):
        if np.random.rand() < mutation_probability:
            mutation_bit = np.random.randint(num_columns)
            offspring[j, mutation_bit] = 1 - offspring[j, mutation_bit]
    return offspring

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

def is_feasible(solution, problem_data):
    matrix_a = np.zeros((len(problem_data["rows"]),len(problem_data["columns"])))
    for row_index,column_indexes in problem_data["rows"].items():
        for column_index in column_indexes:
            matrix_a[row_index,column_index] = 1
    if np.all(np.dot(matrix_a, solution) == 1):
        return True
    else:
        return False

def convert_to_dictionaries(constraint_matrix):
    num_constraints, num_columns = constraint_matrix.shape
    rows = {i: set() for i in range(num_constraints)}
    columns = {j: set() for j in range(num_columns)}
    for i in range(num_constraints):
        for j in range(num_columns):
            if constraint_matrix[i, j] == 1:
                rows[i].add(j)
                columns[j].add(i)
    return rows, columns

def load_data_set(filename):
    with open(filename, 'r') as file:
        num_constraints, num_columns = map(int, file.readline().split())
        constraint_matrix = np.zeros((num_constraints, num_columns), dtype=int)
        column_costs = np.zeros(num_columns, dtype=int)
        for j in range(num_columns):
            data = list(map(int, file.readline().split()))
            column_costs[j] = data[0]
            covered_rows = data[2:]
            for row in covered_rows:
                constraint_matrix[row - 1, j] = 1
    return constraint_matrix, column_costs


if __name__ == "__main__":
    filename = "./data/set_cover/sppnw41.txt"
    constraint_matrix, column_costs = load_data_set(filename)
    rows_dict, columns_dict = convert_to_dictionaries(constraint_matrix)
    problem_data = {}
    problem_data["rows"] = rows_dict
    problem_data["columns"] = columns_dict
    problem_data["cost"] = column_costs
    print("row count:", len(rows_dict))
    print("column count:", len(columns_dict))
    best_solution, best_cost = genetic_algorithm_set_partition(constraint_matrix, column_costs, max_iterations=200, population_size=1000, constraint_violation_penalty=100000)
    print("constraint satisfied :", is_feasible(best_solution, problem_data))
