import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import sys
import time
from tqdm import tqdm
import os



class MaxSatGA:
    def __init__(self, clauses, num_vars, weights, k=0, population_size=10000, pc=0.9, pm1=0.05, pm2=0.05, pm3=0.1, max_generations=10, time_budget=10, penalty=10, debug=False):
        self.clauses = clauses
        self.num_vars = num_vars
        self.weights = weights
        self.k = k  # Number of constraints to learn
        self.population_size = population_size
        self.pc = pc  # Crossover probability
        self.pm1 = pm1  # Hardness mutation probability
        self.pm2 = pm2  # Weight mutation probability
        self.pm3 = pm3  # Literal mutation probability
        self.max_generations = max_generations
        self.time_budget = time_budget
        self.penalty = penalty
        self.debug = debug
        self.population = self.initialize_population()
        # print("population", self.population)

    def initialize_population(self):
        return np.random.randint(0, 2, (self.population_size, self.num_vars))

    def evaluate_population(self):
        return [self.evaluate_individual(ind) for ind in self.population]

    def evaluate_individual(self, individual):
        satisfied, unsatisfied = 0, 0
        for clause, weight in zip(self.clauses, self.weights):
            # print(clause)
            # print(individual)
            satisfied = satisfied+is_clause_satisfied(clause, individual )
            unsatisfied = len(self.clauses)-satisfied
            # if any((int(lit) > 0 and lit in individual) or (-int(lit) in individual) for lit in clause):
                # satisfied += weight
            # else:
            #     unsatisfied += 1
        return satisfied - (self.penalty * unsatisfied)
    
    def crowding_selection(self, fitness_values):
        selected = []
        for _ in range(self.population_size // 2):
            p1, p2 = random.sample(range(self.population_size), 2)
            selected.append(p1 if fitness_values[p1] > fitness_values[p2] else p2)
        return [self.population[i] for i in selected]

    def clause_crossover(self, parent1, parent2):
        # print(parent1, parent2)
        parent1 = ''.join(map(str, parent1)) #Convert numpy.int64 to str
        parent2 = ''.join(map(str, parent2)) #Convert numpy
        crossover_point = random.randint(1, self.num_vars - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        child = np.array([int(digit) for digit in child])
        # print(parent1, parent2, child)
        return child
    
    def mutate_hardness(self, individual):
        if random.random() < self.pm1:
            # print(individual, self.k)
            idx = random.randint(0, self.k-1)
            val = individual[idx]
            individual[idx]  = abs(val-1)
            # print(individual)
        return individual

    # def mutate_weight(self, individual):
    #     if random.random() < self.pm2:
    #         idx = random.randint(0, self.k-1)
    #         self.weights[idx] = random.uniform(0, 1)
    #     return individual

    # def mutate_literal(self, individual):
    #     if random.random() < self.pm3:
    #         idx = random.randint(0, self.k-1)
    #         # print(individual, idx, self.k)
    #         individual[idx] = -individual[idx]
    #     return individual

    def optimize(self):
        start_time = time.time()
        best_solution, best_fitness = None, float('-inf')
        count_generation = 0
        for generation in range(self.max_generations):
            if time.time() - start_time > self.time_budget:
                break
            fitness_values = self.evaluate_population()
            best_idx = np.argmax(fitness_values)
            if fitness_values[best_idx] > best_fitness:
                best_fitness, best_solution = fitness_values[best_idx], self.population[best_idx]
            
            new_population = []
            while len(new_population) < self.population_size:
                i1, i2 = random.sample(range(self.population_size), 2)
                parent1, parent2 = self.population[i1], self.population[i2]
                if random.random() < self.pc:
                    offspring = self.clause_crossover(parent1, parent2)
                else:
                    offspring = random.choice([parent1, parent2])
                offspring = self.mutate_hardness(offspring)
                # offspring = self.mutate_weight(offspring)
                # offspring = self.mutate_literal(offspring)
                new_population.append(offspring)
            
            self.population = new_population
            if self.debug:
                elapsed_time = time.time() - start_time
                print(f'Generation {generation}: Best fitness {best_fitness}, Elapsed Time: {elapsed_time:.2f}s')
        
        runtime = (count_generation + 1) * self.population_size
        return best_solution, best_fitness, runtime


def load_maxsat(wdimacs_file):
    # with open(filename, 'r') as file:
    #     lines = file.readlines()
    # num_vars, num_clauses = map(int, lines[0].split())
    # clauses = []
    # weights = []
    # for line in lines[1:]:
    #     parts = list(map(int, line.split()))
    #     weights.append(parts[0])
    #     clauses.append(parts[1:-1])  # Ignore trailing 0
    clauses = load_maxsat_instance(wdimacs_file)
    num_vars = max(abs(int(literal)) for clause in clauses for literal in map(int, clause.split()[1:-1]))
    weights  = [1]*num_vars
    return clauses, num_vars, weights

def is_clause_satisfied(clause_str, assignment_str):
    # print(clause_str, assignment_str)
    clause = list(map(int, clause_str.split()))
    assignment = list(map(int, assignment_str))
    literals = clause[1:-1]
    
    for literal in literals:
        var_index = abs(int(literal)) - 1
        is_positive = literal > 0
        if (is_positive and assignment[var_index] == 1) or (not is_positive and assignment[var_index] == 0):
            return 1  
    return 0 

def task1(clause_str, assignment_str):
    # print(clause_str, assignment_str)
    clause = list(map(float, clause_str.split()))
    assignment = list(map(int, assignment_str))
    literals = clause[1:-1]
    
    for literal in literals:
        var_index = abs(int(literal)) - 1
        is_positive = literal > 0
        if (is_positive and assignment[var_index] == 1) or (not is_positive and assignment[var_index] == 0):
            return 1  
    return 0 

def count_satisfied_clauses(wdimacs_file, assignment_str):
    satisfied_count = 0
    with open(wdimacs_file, 'r') as file:
        for line in file:
            if line.startswith("c") or line.startswith("p"):
                continue  
            if is_clause_satisfied(line.strip(), assignment_str):
                satisfied_count += 1
    return satisfied_count

def load_maxsat_instance(wdimacs_file):
    clauses = []
    with open(wdimacs_file, 'r') as file:
        for line in file:
            if line.startswith("c") or line.startswith("p"):
                continue
            clauses.append(line.strip())
    return clauses

# if __name__ == "__main__":
#     # filename = "scpcyc06_maxsat.wcnf"
#     filename = "t4pm3-6666.spn.wcnf"
#     # filename = "rev66-6.wcnf"
#     # filename = "test.wcnf"
#     clauses, num_vars, weights = load_maxsat(filename)
#     print("total clauses", len(clauses))
#     solver = MaxSatGA(clauses, num_vars, weights, k=num_vars, population_size=20, pc=0.9, pm1=0.1, pm2=0.1, pm3=0.1, max_generations=100, cutoff_time=6)
#     best_solution, best_fitness = solver.optimize()
#     # best_solution, best_fitness = genetic_algorithm_maxsat(clauses, num_vars, weights, max_iterations=10, penalty=0)
#     # print("Best Solution:", best_solution)
#     print("Max Satisfied Weight:", best_fitness)
#     print("satisfied clauses", count_satisfied_clauses(filename, ''.join(map(str, best_solution))))

fixed_mutation_rate = 0.9
fixed_crossover_rate = 0.9
fixed_pop_size = 20

def run_experiments(wdimacs_file, time_budget=2, repetitions=100):
    # Parameters to test
    pop_sizes = [10,25, 50, 75, 100,]
    mutation_rates = [0.1, 0.25, 0.5, 0.75, 0.9]
    crossover_rates = [0.1, 0.25, 0.5, 0.75, 0.9]

    # Results dictionary to store scores for each parameter setting
    results = {"pop_size": {}, "mutation_rate": {}, "crossover_rate": {}}
    
    # Load the MAXSAT problem instance
    clauses, num_vars, weights = load_maxsat(wdimacs_file)
    
    # # Fixed parameters
    # fixed_mutation_rate = 0.3
    # fixed_crossover_rate = 0.6
    # fixed_pop_size = 20

    # Run experiments for population sizes
    with tqdm(total=len(pop_sizes) * repetitions, desc="Running Pop Size Experiments", unit="iteration", dynamic_ncols=True, leave=True) as pbar:
        for pop_size in pop_sizes:
            scores = []
            for _ in range(repetitions):
                solver = MaxSatGA(
                    clauses, num_vars, weights, k=num_vars, 
                    population_size=pop_size, pc=fixed_crossover_rate, 
                    pm1=fixed_mutation_rate, max_generations=5, time_budget=time_budget
                )
                best_solution, best_fitness, runtime = solver.optimize()
                best_score = count_satisfied_clauses(wdimacs_file, ''.join(map(str, best_solution)))
                scores.append(best_score)
                pbar.update(1)
            results["pop_size"][pop_size] = scores  # Store scores for each pop_size

    # Run experiments for mutation rates
    with tqdm(total=len(mutation_rates) * repetitions, desc="Running Mutation Rate Experiments", unit="iteration", dynamic_ncols=True, leave=True) as pbar:
        for mutation_rate in mutation_rates:
            scores = []
            for _ in range(repetitions):
                solver = MaxSatGA(
                    clauses, num_vars, weights, k=num_vars, 
                    population_size=fixed_pop_size, pc=fixed_crossover_rate, 
                    pm1=mutation_rate, max_generations=5, time_budget=time_budget
                )
                best_solution, best_fitness, runtime = solver.optimize()
                best_score = count_satisfied_clauses(wdimacs_file, ''.join(map(str, best_solution)))
                scores.append(best_score)
                pbar.update(1)
            results["mutation_rate"][mutation_rate] = scores  # Store scores for each mutation_rate

    # Run experiments for crossover rates
    with tqdm(total=len(crossover_rates) * repetitions, desc="Running Crossover Rate Experiments", unit="iteration", dynamic_ncols=True, leave=True) as pbar:
        for crossover_rate in crossover_rates:
            scores = []
            for _ in range(repetitions):
                solver = MaxSatGA(
                    clauses, num_vars, weights, k=num_vars, 
                    population_size=fixed_pop_size, pc=crossover_rate, 
                    pm1=fixed_mutation_rate, max_generations=5, time_budget=time_budget
                )
                best_solution, best_fitness, runtime = solver.optimize()
                best_score = count_satisfied_clauses(wdimacs_file, ''.join(map(str, best_solution)))
                scores.append(best_score)
                pbar.update(1)
            results["crossover_rate"][crossover_rate] = scores  # Store scores for each crossover_rate

    return results, wdimacs_file

def plot_results(results, wdimacs_file):


    # Create a directory to save plots if it doesn't exist
    output_dir = 'experiment_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Population size boxplot
    pop_sizes = list(results["pop_size"].keys())
    pop_scores = list(results["pop_size"].values())
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(pop_scores, labels=pop_sizes)
    plt.title(f"Effect of Population Size on MAXSAT: {wdimacs_file}")
    plt.xlabel("Population Size")
    plt.ylabel("Number of Satisfied Clauses")
    plt.grid(False)

    # Add the fixed parameters to the legend
    plt.legend([f"Fixed Mutation Rate: {fixed_mutation_rate}, Fixed Crossover Rate: {fixed_crossover_rate}"], loc='upper right')

    # Save the population size plot
    pop_size_filename = os.path.join(output_dir, f"{wdimacs_file}_pop_size.png")
    plt.savefig(pop_size_filename)
    plt.close()  # Close the figure to avoid overlap

    # Mutation rate boxplot
    mutation_rates = list(results["mutation_rate"].keys())
    mutation_scores = list(results["mutation_rate"].values())

    plt.figure(figsize=(10, 6))
    plt.boxplot(mutation_scores, labels=mutation_rates)
    plt.title(f"Effect of Mutation Rate on MAXSAT: {wdimacs_file}")
    plt.xlabel("Mutation Rate")
    plt.ylabel("Number of Satisfied Clauses")
    plt.grid(False)

    # Add the fixed parameters to the legend
    plt.legend([f"Fixed Population Size: {fixed_pop_size}, Fixed Crossover Rate: {fixed_crossover_rate}"], loc='upper right')

    # Save the mutation rate plot
    mutation_rate_filename = os.path.join(output_dir, f"{wdimacs_file}_mutation_rate.png")
    plt.savefig(mutation_rate_filename)
    plt.close()

    # Crossover rate boxplot
    crossover_rates = list(results["crossover_rate"].keys())
    crossover_scores = list(results["crossover_rate"].values())

    plt.figure(figsize=(10, 6))
    plt.boxplot(crossover_scores, labels=crossover_rates)
    plt.title(f"Effect of Crossover Rate on MAXSAT: {wdimacs_file}")
    plt.xlabel("Crossover Rate")
    plt.ylabel("Number of Satisfied Clauses")
    plt.grid(False)

    # Add the fixed parameters to the legend
    plt.legend([f"Fixed Population Size: {fixed_pop_size}, Fixed Mutation Rate: {fixed_mutation_rate}"], loc='upper right')

    # Save the crossover rate plot
    crossover_rate_filename = os.path.join(output_dir, f"{wdimacs_file}_crossover_rate.png")
    plt.savefig(crossover_rate_filename)
    plt.close()

    print(f"Plots saved in {output_dir} directory.")

if __name__ == "__main__":
    args = sys.argv
    if "-clause" in args:
        clause_input = args[args.index("-clause") + 1]
        assignment_input = args[args.index("-assignment") + 1]
        print(task1(clause_input, assignment_input))
    elif "-wdimacs" in args and "-assignment" in args:
        wdimacs_file = args[args.index("-wdimacs") + 1]
        assignment_input = args[args.index("-assignment") + 1]
        print(count_satisfied_clauses(wdimacs_file, assignment_input))
    elif "-wdimacs" in args and "-time_budget" in args and "-repetitions" in args:
        wdimacs_file = args[args.index("-wdimacs") + 1]
        time_budget = int(args[args.index("-time_budget") + 1])
        repetitions = int(args[args.index("-repetitions") + 1])
        for i in range(repetitions):
            clauses, num_vars, weights = load_maxsat(wdimacs_file)
            # print("total clauses", len(clauses))
            solver = MaxSatGA(clauses, num_vars, weights, k=num_vars, population_size=1000, pc=0.9, pm1=0.1, pm2=0.1, pm3=0.1, max_generations=3, time_budget=6)
            best_solution, best_fitness, runtime = solver.optimize()
            best_score = count_satisfied_clauses(wdimacs_file, ''.join(map(str, best_solution)))
            # runtime, best_score, best_solution = evolutionary_algorithm(wdimacs_file, time_budget)
            print(runtime, best_score, ''.join(map(str, best_solution)))
    elif "-experiment" in args:
        wdimacs_file = args[args.index("-experiment") + 1]
        results, wdimacs_file = run_experiments(wdimacs_file)
        plot_results(results, wdimacs_file)

