import random


def ranking_replacement(population, fitness_values, violation_values, child_fitness, child_violation):
    G1 = []
    G2 = []
    G3 = []
    G4 = []

    for i in range(len(population)):
        if fitness_values[i] >= child_fitness and violation_values[i] >= child_violation:
            G1.append(i)
        elif fitness_values[i] < child_fitness and violation_values[i] >= child_violation:
            G2.append(i)
        elif fitness_values[i] >= child_fitness and violation_values[i] < child_violation:
            G3.append(i)
        else:
            G4.append(i)
    if G1:
        worst_index = max(G1, key=lambda index: (fitness_values[index], violation_values[index]))
        return worst_index
    elif G2:
        worst_index = max(G2, key=lambda index: violation_values[index])
        return worst_index
    elif G3:
        worst_index = max(G3, key=lambda index: fitness_values[index])
        return worst_index
    elif G4:
        import random
        return random.choice(G4)

    else:
      return 0