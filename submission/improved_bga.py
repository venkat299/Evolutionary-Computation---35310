import numpy as np
import random
import matplotlib.pyplot as plt


def genetic_algorithm_set_partition(constraint_matrix, column_costs, max_iterations=100, population_size=200, constraint_violation_penalty=10000):
    
    num_columns = constraint_matrix.shape[1]
    rows_dict, columns_dict = convert_to_dictionaries(constraint_matrix)
    problem_data = {"rows": rows_dict, "columns": columns_dict, "cost": column_costs}
    population = initialize_population_heuristic(population_size, problem_data)
    
    best_solution = None
    best_fitness = float('inf')
    best_violation = float('inf')

    for iteration in range(max_iterations):
        # Evaluate the fitness and constraint violations of the current population.
        fitness_values, violation_values = evaluate_population(population, constraint_matrix, column_costs, constraint_violation_penalty)
        
        # Find the best solution in the current population.
        for i in range(population_size):
            # If a solution is feasible (no violations) and better than the current best, update.
            if violation_values[i] == 0 and (best_violation > 0 or fitness_values[i] < best_fitness):
                best_fitness = fitness_values[i]
                best_solution = population[i]
                best_violation = violation_values[i]
            # If the current best is infeasible, look for better infeasible solutions.
            elif best_violation > 0 and violation_values[i] < best_violation:
                best_violation = violation_values[i]
                best_solution = population[i]
                best_fitness = fitness_values[i]
                
        # Select parents for crossover using tournament selection.
        parents = perform_tournament_selection(population, fitness_values, tournament_size=3)
        # Perform uniform crossover to generate offspring.
        offspring = perform_uniform_crossover(parents, population_size, num_columns)
        # Apply mutation to the offspring.
        offspring = perform_mutation(offspring, num_columns)
        
        # Apply heuristic improvement to each child, and remove duplicate children.
        new_offspring = []
        for child in offspring:
            improved_child = heuristic_improvement_operator(child, problem_data)
            if not any(np.array_equal(improved_child, existing_member) for existing_member in population):
                new_offspring.append(improved_child)
        
        # Replace the worst individuals in the population with the new offspring.
        for child in new_offspring:
            fitness, violation = evaluate_individual(child, constraint_matrix, column_costs, constraint_violation_penalty)
            replace_index = ranking_replacement(population, fitness_values, violation_values)
            population[replace_index] = child
            fitness_values[replace_index] = fitness
            violation_values[replace_index] = violation

            # Update the best solution if a better solution is found.
            if (violation == best_violation == 0 and fitness < best_fitness) or (best_violation > 0 and violation < best_violation):
                best_fitness = fitness
                best_solution = child
                best_violation = violation
        
        # Print the progress of the algorithm.
        print(f'generation:{iteration}, violation:{best_violation}, cost min:{best_fitness}, cost max:{max(fitness_values)}, std:{round(np.std(fitness_values))}')
        
    
    # Print the final results.
    print(f"Minimum cost found by GA: {best_fitness}")
    print(f"Total constraint violations: {best_violation}")
    return best_solution, best_fitness

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
    total_costs = np.sum(population * column_costs, axis=1)
    constraint_violations = np.sum((np.dot(constraint_matrix, population.T) - 1) ** 2, axis=0)
    fitness_values = total_costs + constraint_violation_penalty * constraint_violations
    return fitness_values, total_costs, constraint_violations

def perform_mutation(offspring, num_columns):
    mutation_probability = random.uniform(1/num_columns, 0.8)
    for j in range(len(offspring)):
        if np.random.rand() < mutation_probability:
            mutation_bit = np.random.randint(num_columns)
            offspring[j, mutation_bit] = 1 - offspring[j, mutation_bit]
    return offspring

def initialize_population_heuristic(population_size, problem_data):
    population_size = max(1, population_size)
    rows = problem_data["rows"]
    columns = problem_data["columns"]
    population = []
    row_indices = list(rows.keys())
    for _ in range(population_size):
        solution_set = set()
        uncovered_rows = set(row_indices)
        while uncovered_rows:
            row_index = random.choice(list(uncovered_rows))
            valid_columns = [column_index for column_index in rows[row_index] if not columns[column_index].intersection(set(row_indices) - uncovered_rows)]
            if valid_columns:
                column_index = random.choice(valid_columns)
                solution_set.add(column_index)
                uncovered_rows.difference_update(columns[column_index])
            else:
                uncovered_rows.remove(row_index)
        solution = [0] * len(problem_data["columns"])
        for index in solution_set:
            solution[index] = 1
        population.append(np.array(solution))
    return np.array(population)

def evaluate_population(population, constraint_matrix, column_costs, constraint_violation_penalty):
    fitness_values = np.zeros(population.shape[0])
    violation_values = np.zeros(population.shape[0])
    for i, individual in enumerate(population):
        fitness_values[i], violation_values[i] = evaluate_individual(individual, constraint_matrix, column_costs, constraint_violation_penalty)
    return fitness_values, violation_values

def evaluate_individual(individual, constraint_matrix, column_costs, constraint_violation_penalty):
    total_cost = np.sum(individual * column_costs)
    constraint_violation = np.sum((np.dot(constraint_matrix, individual.T) - 1) ** 2)
    fitness = total_cost + constraint_violation_penalty * constraint_violation
    return fitness, constraint_violation

def ranking_replacement(population, fitness_values, violation_values):
    indices = np.argsort(violation_values)
    feasible_indices = indices[violation_values[indices] == 0]
    infeasible_indices = indices[violation_values[indices] > 0]
    
    if len(feasible_indices) > 0:
        worst_feasible_index = feasible_indices[-1]
        return worst_feasible_index
    else:
        worst_infeasible_index = infeasible_indices[-1]
        return worst_infeasible_index

def heuristic_improvement_operator(solution, problem_data):
    rows = problem_data["rows"]
    columns = problem_data["columns"]
    column_costs = problem_data["cost"]
    solution_set = set(i for i, val in enumerate(solution) if val == 1)
    row_cover_counts = {i: len(rows[i].intersection(solution_set)) for i in rows}
    
    # DROP procedure
    temp_set = solution_set.copy()
    while temp_set:
        column_j = random.choice(list(temp_set))
        temp_set.remove(column_j)
        if any(row_cover_counts[i] >= 2 for i in columns[column_j]):
            solution_set.remove(column_j)
            for i in columns[column_j]:
                row_cover_counts[i] -= 1
    
    # ADD procedure
    uncovered_rows = {i for i, count in row_cover_counts.items() if count == 0}
    temp_uncovered = uncovered_rows.copy()
    while temp_uncovered:
        row_i = random.choice(list(temp_uncovered))
        temp_uncovered.remove(row_i)
        
        valid_columns = []
        for col_j in rows[row_i]:
            if columns[col_j].issubset(uncovered_rows):
                valid_columns.append(col_j)
        
        if valid_columns:
            best_column = min(valid_columns, key=lambda col_j: column_costs[col_j] / len(columns[col_j]))
            solution_set.add(best_column)
            for i in columns[best_column]:
                row_cover_counts[i] += 1
                if i in uncovered_rows:
                    uncovered_rows.remove(i)
                if i in temp_uncovered:
                    temp_uncovered.remove(i)
    
    improved_solution = np.zeros_like(solution)
    for col_index in solution_set:
        improved_solution[col_index] = 1
    return improved_solution

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
    filename = "./data/set_cover/sppnw43.txt"
    constraint_matrix, column_costs = load_data_set(filename)
    rows_dict, columns_dict = convert_to_dictionaries(constraint_matrix)
    problem_data = {}
    problem_data["rows"] = rows_dict
    problem_data["columns"] = columns_dict
    problem_data["cost"] = column_costs
    print("row count:", len(rows_dict))
    print("column countprint(f'generation:{iteration}, cost:{best_fitness}, violation:{best_violation}, variance:{round(np.std(fitness_values))}'):", len(columns_dict))
    best_solution, best_cost = genetic_algorithm_set_partition(constraint_matrix, column_costs, max_iterations=100, population_size=100, constraint_violation_penalty=20000)
    print("constraint satisfied :", is_feasible(best_solution, problem_data))