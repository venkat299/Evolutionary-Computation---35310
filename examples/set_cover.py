import random

from matplotlib.artist import get
from src.metaheuristics import simulated_annealing as sa

import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        coverage.append(elements)
    
    return num_rows, num_columns, costs, coverage

# generate initial solution
def initial_solution(num_rows, num_columns, coverage):
    solution = set()
    uncovered_rows = set(range(1, num_rows + 1))
    
    while uncovered_rows:
        col = random.randint(0, num_columns - 1)
        solution.add(col)
        uncovered_rows -= set(coverage[col])

    return solution

# calculate cost
def solution_cost(costs, solution):
    return sum(costs[col] for col in solution)

# Check feasibility of a solution
def is_valid(solution, coverage, num_rows):
    covered = set()
    for col in solution:
        covered.update(coverage[col])
    return covered == set(range(1, num_rows + 1))

# Generate a neighbor solution
def get_neighbor(solution, coverage, num_columns, num_rows):
    new_solution = solution.copy()
    if random.random() < 0.5 and len(new_solution) > 1:  # Remove a column
        new_solution.remove(random.choice(list(new_solution)))
    else:  # Add a column
        new_solution.add(random.randint(0, num_columns - 1))

    if is_valid(new_solution,coverage, num_rows):
        return new_solution
    return solution  # If invalid, return the original

def set_cover(num_rows, num_columns, costs, coverage):
    
    initial_sol = initial_solution(num_rows, num_columns, coverage)
    
    def obj_fn(solution):
        return (solution_cost(costs, solution))
    
    def neighbour(curr_sol):
        return get_neighbor(curr_sol, coverage, num_columns, num_rows)
    
    sa_params = {
        "obj_fn": obj_fn,
        "get_neighbour": neighbour,
        "init_sol":initial_sol,
        "init_temp": 100,
        "cooling_rate": 0.99995,
        # "min_temp": 0.1,
        "epochs": 1000
    }
    
    epochs = sa_params["epochs"]

    # Run Simulated Annealing
    best_solution, best_cost_history, cost_history, temperature_history = sa.simulated_annealing(**sa_params)
    
    fig, ax1 = plt.subplots()
    # ax1.set_xlabel("Epoch")
    # ax1.set_ylabel("Objective Function Value", color='tab:blue')
    # ax1.plot(range(epochs), best_cost, color='tab:blue', label="Best obj value")max(cost_history)
    # ax1.plot(range(epochs), cost_history, color='tab:cyan', linestyle='dashed', label="Current obj value")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Temperature", color='tab:red')
    # ax2.plot(range(epochs), temperature_history, color='tab:red', linestyle='dashed', label="Temperature")
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    fig.tight_layout()
    # plt.title("Simulated Annealing: Epoch vs Objective Function & Temperature")
    ax1.axhline(y=cost_history[-1], color='green', linestyle='dotted', label=f'Final Obj Value: {cost_history[-1]:.2f}')
    # ax1.legend()
    
    ax1.set_xlim(0, len(cost_history))
    # ax1.set_ylim(min(cost_history) - 0.05*min(cost_history), max(cost_history) + 0.05*max(cost_history))
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Best Cost Found")
    
    # ax2.set_xlim(0, len(tem))
    ax2.set_ylim(min(temperature_history) - 0.05*min(temperature_history), max(temperature_history) + 0.05*max(temperature_history))


    ax1.set_title("Simulated Annealing: Best Cost over Iterations")

    line_best_cost, = ax1.plot([], [], lw=2, color="blue", label="Best Cost")
    line_current_cost, = ax1.plot([], [], lw=2, color="tab:cyan",linestyle='dashed', label="Current Cost")
    line_temperature, = ax2.plot([], [], lw=2, color="red", label="Temperature")
    ax1.legend()
    # ax2.legend()

    # Animation function
    def update(frame):
        line_best_cost.set_data(range(frame + 1), best_cost_history[:frame + 1])
        line_current_cost.set_data(range(frame + 1), cost_history[:frame + 1])
        line_temperature.set_data(range(frame + 1), temperature_history[:frame + 1])
        ax1.relim()
        ax1.autoscale_view()
        # ax2.relim()
        # ax2.autoscale_view()
        return line_best_cost, line_current_cost, line_temperature

    # Animate
    ani = animation.FuncAnimation(fig, update, frames=len(best_cost_history), interval=50, blit=True, repeat=False)

    plt.show()

    # Print the best solution
    print("Selected Columns:", sorted(best_solution))
    print("Minimum Cost:", best_cost_history[-1])


if __name__ == "__main__":

    # Example usage:
    file_path = './data/set_cover/sppnw41.txt'  # Replace with the actual file path
    num_rows, num_columns, costs, coverage = read_set_cover_data(file_path)

    # Print the results
    print("num_rows:", num_rows)
    print("num_columns:", num_columns)
    print("costs:", costs)
    print("coverage:", coverage)
    
    set_cover(num_rows, num_columns, costs, coverage)
    
    