import random


def initialize_population_heuristic(population_size, problem_data):
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




def is_feasible(solution, problem):
    covered_rows = set()  
    row_counts = {}  
    
    
    for i, selected in enumerate(solution):
        if selected:  
            covered_rows.update(problem['columns'][i])
            for row in problem['columns'][i]:
                if row in row_counts:
                    row_counts[row] += 1
                else:
                    row_counts[row] = 1

    for row in problem['rows'].keys():
        if row not in row_counts or row_counts[row] != 1:
            return False  
    return len(covered_rows) == len(problem['rows'])  



def invert_dict_of_sets(d):
    inverted = {}
    for key, value_set in d.items():
        for v in value_set:
            if v not in inverted:
                inverted[v] = set()
            inverted[v].add(key)
    return inverted




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
        coverage.append([i-1 for i in elements])
    columns = dict(enumerate(coverage))
    for item in columns:
        columns[item] = set(columns[item])
    rows = invert_dict_of_sets(columns)
    return  rows, columns, costs


if __name__ == "__main__":

    file_path = './sppnw41.txt'  
    rows, columns, costs = read_set_cover_data(file_path)
    
    # Print the results
    print("num_rows:", len(rows))
    print("num_columns:", len(columns))
    # print("costs:", costs)
    # print("rows:", rows)
    # print("columns:", columns)
    
    problem = {}
    problem["rows"] = rows  
    problem["columns"] = columns
    problem["cost"]= costs
    
    solution = initialize_population_heuristic(1, problem)
    print(solution)