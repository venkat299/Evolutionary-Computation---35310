import math
import random

def simulated_annealing(obj_fn, get_neighbour, init_sol, epochs=1000, init_temp=100, cooling_rate=0.995):
    curr_sol = init_sol   # Initial solution, objective function value (energy).
    curr_obj = obj_fn(curr_sol)
    best_sol = curr_sol # Initial “best” solution
    best_obj = curr_obj
    temp = init_temp
    
    temp_values = []
    obj_values = []
    curr_obj_values = []
    
    for k in range(epochs):
        new_sol = get_neighbour(curr_sol) # Pick some neighbour.
        new_obj = obj_fn(new_sol) # Compute its objective function value.
        print(f'epoch {k}, temp: {round(temp,1)}, best :{round(best_obj,1)}, new:{round(new_obj)},  ') #sol:{best_sol}
        
        if new_obj < curr_obj or random.random() < math.exp((curr_obj - new_obj) / temp):
            curr_sol = new_sol
            curr_obj = new_obj
        
        if curr_obj < best_obj:
            best_sol = curr_sol
            best_obj = curr_obj
        
        obj_values.append(best_obj)
        curr_obj_values.append(curr_obj)
        temp_values.append(temp)
        temp *= cooling_rate  # Temperature calculation.
    
    return best_sol,  obj_values, curr_obj_values, temp_values
