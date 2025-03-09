import random
import time
import csv
import multiprocessing
from simulated_annealing import sa
from standard_bga_with_numpy import bga
from improved_bga_with_numpy import ibga
from improved_bga_stochastic_ranking_with_numpy import ibgasr


def run_benchmark(file_path, algorithm, num_runs=30, num_processes=4):
    results = []
    times = []
    feasible_flag = []
    l = lambda feasible: 1 if feasible else 0

    for run in range(num_runs):
        start_time = time.time()
        best_solution, best_cost, feasible = algorithm(file_path, num_processes)
        feasible_flag.append(l(feasible))
        end_time = time.time()
        results.append(best_cost)
        times.append(end_time - start_time)
        print(f'algo:{algorithm.__name__}, run:{run}, best_cost:{best_cost}, feasible:{l(feasible)}' )

    avg_result = sum(results) / num_runs
    std_dev_result = calculate_std(results)
    avg_time = sum(times) / num_runs
    std_dev_time = calculate_std(times)

    return avg_result, std_dev_result, avg_time, std_dev_time, sum(feasible_flag)

def calculate_std(values):
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5

def benchmark_problems(algo_list, problem_files, num_processes=4):

    results_data = []
    for algorithm in algo_list:
        for filename in problem_files:
            avg_result, std_dev_result, avg_time, std_dev_time, feasible_count = run_benchmark(filename, algorithm, num_processes=num_processes)
            problem_name = filename.split("/")[-1]
            print(f"Benchmark: {problem_name}")
            print(f"  Algorithm: {algorithm.__name__}")
            print(f"  Average Result: {avg_result:.2f}, Standard Deviation: {std_dev_result:.2f}")
            print(f"  Average Time: {avg_time:.2f}s, Standard Deviation: {std_dev_time:.2f}s")
            print(f"  feasible count:{feasible_count}")

            results_data.append([problem_name, algorithm.__name__, round(avg_result), round(std_dev_result,3), round(avg_time,3), round(std_dev_time,3)])

    write_results_to_csv(results_data)

def write_results_to_csv(results_data, filename="./benchmark_results.csv"):
    """Writes benchmark results to a CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Problem", "Algorithm", "Avg Result", "Std Dev Result", "Avg Time (s)", "Std Dev Time (s)", "feasible results"])
        writer.writerows(results_data)


if __name__ == "__main__":
    # def my_algorithm(constraint_matrix, column_costs, problem_data, num_processes):
    #     return sa(constraint_matrix, column_costs, problem_data, num_processes)

    problem_files = [
        "./sppnw41.txt",
        "./sppnw42.txt",
        "./sppnw43.txt"
    ]

    benchmark_problems([sa, bga, ibga, ibgasr ], problem_files, num_processes=multiprocessing.cpu_count())
    # 