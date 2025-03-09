import random
import multiprocessing

def genetic_algorithm_set_partition(constraint_matrix, column_costs, max_iterations=100, population_size=200, constraint_violation_penalty=10000, num_processes=4):
    num_columns = len(constraint_matrix[0])
    population = [[random.random() > 0.5 for _ in range(num_columns)] for _ in range(population_size)]
    best_fitness_values = []
    average_fitness_values = []

    for iteration in range(max_iterations):
        fitness_values, best_fitness, total_constraint_violations = calculate_fitness(population, constraint_matrix, column_costs, constraint_violation_penalty, num_processes)
        sorted_indices = sorted(range(len(fitness_values)), key=lambda k: fitness_values[k])
        population = [population[i] for i in sorted_indices]
        best_fitness_values.append(fitness_values[0])
        average_fitness_values.append(sum(fitness_values) / len(fitness_values))
        parents = perform_tournament_selection(population, fitness_values, tournament_size=3)
        offspring = perform_uniform_crossover(parents, population_size, num_columns)
        offspring = perform_mutation(offspring, num_columns)
        population.extend(offspring)
        fitness_values, best_fitness, total_constraint_violations = calculate_fitness(population, constraint_matrix, column_costs, constraint_violation_penalty, num_processes)
        sorted_indices = sorted(range(len(fitness_values)), key=lambda k: fitness_values[k])
        population = [population[i] for i in sorted_indices[:population_size]]
        print(f'generation:{iteration}, violation:{total_constraint_violations[0]}, cost min:{best_fitness[0]}, cost max:{max(best_fitness)}, std:{round(calculate_std(best_fitness))}')
        if iteration == max_iterations - 1:
            print(f"Final iteration reached: {max_iterations}")
    best_solution = population[0]
    best_cost = best_fitness[0]
    best_violations = total_constraint_violations[0]
    print(f"Minimum cost found by GA: {best_cost}")
    print(f"Total constraint violations: {best_violations}")
    return best_solution, best_cost

def perform_uniform_crossover(parents, population_size, num_bits):
    offspring = []
    for i in range(0, population_size - 1, 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]
        crossover_mask = [random.randint(0, 1) for _ in range(num_bits)]
        child1 = [parent1[j] if crossover_mask[j] == 0 else parent2[j] for j in range(num_bits)]
        child2 = [parent2[j] if crossover_mask[j] == 0 else parent1[j] for j in range(num_bits)]
        offspring.extend([child1, child2])
    return offspring

def perform_tournament_selection(population, fitness_values, tournament_size=3):
    selected_parents = []
    for _ in range(len(population)):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
        selected_parents.append(population[winner_idx])
    return selected_parents

def calculate_fitness(population, constraint_matrix, column_costs, constraint_violation_penalty, num_processes):
    population_size = len(population)
    best_fitness = multiprocessing.Array('d', [0.0] * population_size)
    constraint_violations = multiprocessing.Array('i', [0] * population_size)
    fitness_values = multiprocessing.Array('d', [0.0] * population_size)

    def calculate_individual_fitness(start, end):
        for i in range(start, end):
            individual = population[i]
            cost = sum(individual[j] * column_costs[j] for j in range(len(individual)))
            violation = 0
            for row in constraint_matrix:
                constraint_sum = sum(row[j] * individual[j] for j in range(len(individual)))
                violation += (constraint_sum - 1) ** 2
            best_fitness[i] = cost
            constraint_violations[i] = violation
            fitness_values[i] = cost + constraint_violation_penalty * violation

    processes = []
    chunk_size = population_size // num_processes
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size if i < num_processes - 1 else population_size
        process = multiprocessing.Process(target=calculate_individual_fitness, args=(start, end))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    return list(fitness_values), list(best_fitness), list(constraint_violations)

def perform_mutation(offspring, num_columns):
    mutation_probability = random.uniform(1 / num_columns, 0.8)
    for j in range(len(offspring)):
        if random.random() < mutation_probability:
            mutation_bit = random.randint(0, num_columns - 1)
            offspring[j][mutation_bit] = 1 - offspring[j][mutation_bit]
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
    matrix_a = [[0 for _ in range(len(problem_data["columns"]))] for _ in range(len(problem_data["rows"]))]
    for row_index, column_indexes in problem_data["rows"].items():
        for column_index in column_indexes:
            matrix_a[row_index][column_index] = 1
    for row in matrix_a:
        if sum(row[i] * solution[i] for i in range(len(row))) != 1:
            return False
    return True

def convert_to_dictionaries(constraint_matrix):
    num_constraints = len(constraint_matrix)
    num_columns = len(constraint_matrix[0])
    rows = {i: set() for i in range(num_constraints)}
    columns = {j: set() for j in range(num_columns)}
    for i in range(num_constraints):
        for j in range(num_columns):
            if constraint_matrix[i][j] == 1:
                rows[i].add(j)
                columns[j].add(i)
    return rows, columns

def load_data_set(filename):
    with open(filename, 'r') as file:
        num_constraints, num_columns = map(int, file.readline().split())
        constraint_matrix = [[0 for _ in range(num_columns)] for _ in range(num_constraints)]
        column_costs = [0] * num_columns
        for j in range(num_columns):
            data = list(map(int, file.readline().split()))
            column_costs[j] = data[0]
            covered_rows = data[2:]
            for row in covered_rows:
                constraint_matrix[row - 1][j] = 1
    return constraint_matrix, column_costs

def calculate_std(values):
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5



if __name__ == "__main__":
    filename = "./data/set_cover/sppnw42.txt"
    constraint_matrix, column_costs = load_data_set(filename)
    rows_dict, columns_dict = convert_to_dictionaries(constraint_matrix)
    problem_data = {}
    problem_data["rows"] = rows_dict
    problem_data["columns"] = columns_dict
    problem_data["cost"] = column_costs
    print("row count:", len(rows_dict))
    print("column count:", len(columns_dict))
    best_solution, best_cost = genetic_algorithm_set_partition(constraint_matrix, column_costs, max_iterations=100, population_size=1000, constraint_violation_penalty=100000)
    print("constraint satisfied :", is_feasible(best_solution, problem_data))
