import numpy as np
import random


def stochastic_ranking(population, child, child_fitness, child_violation, fitness_values, violation_values, pf):
    population_size = population.shape[0]
    worst_index = np.argmax(violation_values)
    if violation_values[worst_index] == 0:
        worst_index = np.argmax(fitness_values)

    population[worst_index] = child
    fitness_values[worst_index] = child_fitness
    violation_values[worst_index] = child_violation

    for _ in range(population_size):
        for i in range(population_size - 1):
            u = random.uniform(0, 1)
            if (violation_values[i] == violation_values[i + 1] == 0) or (u <= pf):
                if fitness_values[i] > fitness_values[i + 1]:
                    population[[i, i + 1]] = population[[i + 1, i]]
                    fitness_values[[i, i + 1]] = fitness_values[[i + 1, i]]
                    violation_values[[i, i + 1]] = violation_values[[i + 1, i]]
            else:
                if violation_values[i] > violation_values[i + 1]:
                    population[[i, i + 1]] = population[[i + 1, i]]
                    fitness_values[[i, i + 1]] = fitness_values[[i + 1, i]]
                    violation_values[[i, i + 1]] = violation_values[[i + 1, i]]
    return population

def test_stochastic_ranking():
    np.random.seed(42)  # Set seed for reproducibility
    random.seed(42)

    # Create a mock population (size = 5, each individual has 4 genes)
    population_size = 5
    num_genes = 4
    population = np.random.randint(0, 2, (population_size, num_genes))

    # Generate random fitness values and constraint violations
    fitness_values = np.array([10, 20, 30, 40, 50])  # Higher is worse
    violation_values = np.array([0, 1, 0, 2, 1])  # Lower is better

    # New child with some fitness and violation
    child = np.array([1, 0, 1, 1])  # A random new individual
    child_fitness = 15
    child_violation = 0

    # Probability of ranking based on fitness instead of constraint violation
    pf = 0.5

    # Call the stochastic_ranking function
    new_population = stochastic_ranking(population, child, child_fitness, child_violation, fitness_values, violation_values, pf)

    # Assertions
    assert new_population.shape == (population_size, num_genes), "Population size should remain unchanged."
    assert np.any(np.all(new_population == child, axis=1)), "Child should have replaced the worst individual."
    assert np.all(np.diff(fitness_values) >= 0) or np.all(np.diff(violation_values) >= 0), "Sorting should be correct."

    print("Test passed successfully!")

# Run the test
test_stochastic_ranking()
