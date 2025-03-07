
import random
import math
import matplotlib.pyplot as plt


from src.metaheuristics import simulated_annealing as sa
from src.local_search import n_opt as nopt

def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)


def total_distance(tour, cities):
    # print("tour", tour)
    if not all(isinstance(i, int) for i in tour):
        raise TypeError("Tour must be a list of integer indices")
    return sum(distance(cities[int(tour[i])], cities[int(tour[i+1])]) for i in range(len(tour)-1)) + distance(cities[int(tour[-1])], cities[int(tour[0])])


def tsp_simulated_annealing(cities, epochs=10):
    init_tour = list(range(len(cities)))
    random.shuffle(init_tour)
    
    def obj_fn(tour):
        return total_distance(tour, cities)
    
    def neighbour_fn(tour):
        return nopt.two_opt_swap(tour)
    
    best_tour, obj_values, curr_obj_values, temp_values = sa.simulated_annealing(obj_fn, neighbour_fn, init_tour, epochs=epochs)
    
    # plot
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Objective Function Value", color='tab:blue')
    ax1.plot(range(epochs), obj_values, color='tab:blue', label="Best obj value")
    ax1.plot(range(epochs), curr_obj_values, color='tab:cyan', linestyle='dashed', label="Current obj value")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Temperature", color='tab:red')
    ax2.plot(range(epochs), temp_values, color='tab:red', linestyle='dashed', label="Temperature")
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    fig.tight_layout()
    plt.title("Simulated Annealing: Epoch vs Objective Function & Temperature")
    ax1.axhline(y=obj_values[-1], color='green', linestyle='dotted', label=f'Final Obj Value: {obj_values[-1]:.2f}')
    ax1.legend()
    plt.show()
    
    return best_tour, total_distance(best_tour, cities), obj_values



if __name__ == "__main__":
    

    # Example TSP-20 dataset
    cities_20 = [
        (random.uniform(0, 100), random.uniform(0, 100)) for _ in range(20)
    ]
    epochs =1000
    best_tour, best_distance, obj_values = tsp_simulated_annealing(cities_20, epochs=epochs)
    print("Best Tour:", best_tour)
    print("Best Distance:", best_distance)
    

    
