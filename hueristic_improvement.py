import random
import numpy as np

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