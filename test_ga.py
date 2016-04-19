from ga import *
from test_functions import *

# Set GA variables
POP_SIZE = 100
MAX_GENS = 5000
RETAIN_PERCENT = 0.2
MUTATION_PROB = 0.1

# Define test functions, parameters and optimal fitness values
test_func_list = [booth_function, ackley_function, bukin_function]
test_params_list = [booth_params, ackley_params, bukin_params]
opt_fitness_list = [0.0, 0.0, 0.0]

# Set number of runs
RUNS = 30

# Create an empty DataFrame to hold the final results
overall_test_results = pd.DataFrame()

for i, test_func in enumerate(test_func_list):
    # Run for a set number of generations:
    # Percentage of runs to optimal fitness
    """
    Run the GA 30 times with a combination of MAX_GENS and POP_SIZE that equates to 500,000 fitness function evaluations
    such as MAX_GEN=5000, POP_SIZE=100

    Return the percentage of these runs that converged to the optimal solution of x=0
    Calculate the average best fitness and average mean fitness achieved accross the 30 runs.
    """
    # Create GA object with finite number of generations
    finite_ga = BaseGenetic(test_params_list[i], test_func, POP_SIZE, RETAIN_PERCENT, MUTATION_PROB,
                            max_gens=MAX_GENS,
                            opt_fitness=opt_fitness_list[i])

    # Run test
    finite_test_results = test_ga(RUNS, finite_ga, verbose=True)

    overall_test_results = overall_test_results.append(finite_test_results, ignore_index=True)

    # Run until convergence to optimal value:
    # Average number of fitness evaluations for the GA to optimally (or near optimally) converge
    # Average best fitness, Average mean fitness
    """
    Run the GA 30 times and calculate the number of fitness evaluations needed for optimal convergence per run.
    Average this list for a measure of the convergence velocity.

    Calculate the average number of evaluations for optimal fitness divided by the standard deviation
    of the set to get the coeff. of variation or C.V. which is a measure of reliability.

    Substitute near optimal convergence as another measure when optimal convergence is not achieved.
    """
    # Create GA object with infinite (default) number of generations
    infinite_ga = BaseGenetic(test_params_list[i], test_func, POP_SIZE, RETAIN_PERCENT, MUTATION_PROB,
                            opt_fitness=opt_fitness_list[i])

    # Run test
    infinite_test_results = test_ga(RUNS, infinite_ga, verbose=True)

    overall_test_results = overall_test_results.append(infinite_test_results, ignore_index=True)

# Save results to CSV
overall_test_results.to_csv('test results/test_results.csv')

# Print overall results
print '\n\n\n\n'

print tabulate(overall_test_results, headers='keys', tablefmt='psql')

