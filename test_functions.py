from collections import OrderedDict
import numpy as np
import pandas as pd
from tabulate import tabulate

# Ackley Function
# f(0, 0) == 0
# -5 <= x, y >= 5
ackley_params = OrderedDict([
        ('x', [num/10000.0 for num in xrange(-50000, 60000)]),
        ('y', [num/1000.0 for num in xrange(-5000, 6000)])
    ])

def ackley_function(ind, params):
    x = params['x'][ind[0]]
    y = params['y'][ind[1]]

    if x == 0.0 and y == 0.0:
        return 0.0

    else:
        sum1 = x**2 + y**2
        sum2 = np.cos(2*np.pi*x) + np.cos(2*np.pi*y)
        term1 = -20 * np.exp(-0.2*np.sqrt(sum1/2))
        term2 = -np.exp(sum2/2)
        s = term1 + term2 + 20 + np.exp(1)
        return s

# Beale Function
# f(3, 0.5) == 0
# -4.5 <= x, y >= 4.5
beale_params = OrderedDict([
    ('x', [num / 10.0 for num in xrange(-45, 55)]),
    ('y', [num / 10.0 for num in xrange(-45, 55)])
])

def beale_function(ind, params):
    x = params['x'][ind[0]]
    y = params['y'][ind[1]]

    if x == 3.0 and y == 0.5:
        return 0.0

    else:
        term1 = (1.5 - x + x*y)**2
        term2 = (2.25 - x + x*y**2)**2
        term3 = (2.625 - x + x*y**3)**2
        s = term1 + term2 + term3
        return s

# Goldstein-Price Function
# f(0, -1) == 3
# -2 <= x, y >= 2
goldstein_params = OrderedDict([
    ('x', [num / 100.0 for num in xrange(-200, 300)]),
    ('y', [num / 100.0 for num in xrange(-200, 300)])
])

def goldstein_function(ind, params):
    x = params['x'][ind[0]]
    y = params['y'][ind[1]]

    if x == 0.0 and y == -1.0:
        return 3.0

    else:
        term1 = (x + y + 1)**2
        term2 = 19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2
        term3 = (2*x - 3*y)**2
        term4 = 18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2
        s = (1 + (term1 * term2)) * (30 + (term3 * term4))
        return s

# Booth Function
# f(1, 3) == 0
# -10 <= x, y >= 10
booth_params = OrderedDict([
    ('x', [num / 10.0 for num in xrange(-100, 110)]),
    ('y', [num / 10.0 for num in xrange(-100, 110)])
])

def booth_function(ind, params):
    x = params['x'][ind[0]]
    y = params['y'][ind[1]]

    if x == 1.0 and y == 3.0:
        return 0.0

    else:
        term1 = (x + 2*y - 7)**2
        term2 = (2*x + y - 5)**2
        s = term1 + term2
        return s

# Bukin Function N.6
# f(-10, 1) == 0
# -15 <= x >= -5
# -3 <= x >= 3
bukin_params = OrderedDict([
    ('x', [num / 10.0 for num in xrange(-150, -60)]),
    ('y', [num / 10.0 for num in xrange(-30, 40)])
])

def bukin_function(ind, params):
    x = params['x'][ind[0]]
    y = params['y'][ind[1]]

    if x == -10.0 and y == 1.0:
        return 0.0

    else:
        term1 = 100 * np.sqrt(abs(y-0.01*x**2))
        term2 = 0.01 * abs(x + 10)
        s = term1 + term2
        return s

# Create function to test GA objects
def test_ga(runs, ga_obj, verbose=False):
    test_results = pd.DataFrame()

    print '_' * 200
    print('RUN 1 ({0:s}, max_gens={1:0.0f})'.format(ga_obj.obj_func.__name__, ga_obj.max_gens))
    for run in xrange(1, runs + 1):
        avg_fitnesses, min_fitnesses, max_fitnesses, best_solutions, gens_created = ga_obj.optimise(verbose=verbose)

        if min_fitnesses[-1] == ga_obj.opt_fitness:
            opt_run = True
        else:
            opt_run = False

        run_result = pd.DataFrame({
            'ga_type': [ga_obj.__class__.__name__],
            'test_func': [ga_obj.obj_func.__name__],
            'poss_permutations': [ga_obj.poss_permutations],
            'run': [run],
            'best_fitness': [min_fitnesses[-1]],
            'avg_fitness': [avg_fitnesses[-1]],
            'opt_fitness': [ga_obj.opt_fitness],
            'best_solution': [best_solutions[-1]],
            'MAX_GEN': [ga_obj.max_gens],
            'POP_SIZE': [ga_obj.pop_size],
            'RETAIN_PERCENT': [ga_obj.retain_percent],
            'MUTATION_PROB': [ga_obj.mutation_prob],
            'fitness_evaluations': [gens_created * ga_obj.pop_size],
            'opt_run': [opt_run]
        })

        test_results = test_results.append(run_result, ignore_index=True)
        if run+1 <= runs:
            print '_' * 100
            print('RUN {0:d} ({1:s}, max_gens={2:0.0f})'.format(run+1, ga_obj.obj_func.__name__, ga_obj.max_gens))

        count_opt_runs = 0
    for run in test_results.best_fitness:
        if run == test_results.opt_fitness[0]:
            count_opt_runs += 1

    if ga_obj.max_gens == np.inf:
        avg_evals = test_results.fitness_evaluations.mean()
        std_evals = test_results.fitness_evaluations.std()
        cv = avg_evals / std_evals
    else:
        avg_evals = np.nan
        std_evals = np.nan
        cv = np.nan

    opt_run_perc = (count_opt_runs / float(runs)) * 100
    avg_best_fitness = test_results.best_fitness.mean()
    avg_mean_fitness = test_results.avg_fitness.mean()

    test_results['avg_evals'] = [avg_evals] * len(test_results)
    test_results['std_evals'] = [std_evals] * len(test_results)
    test_results['cv'] = [cv] * len(test_results)
    test_results['opt_run_perc'] = [opt_run_perc] * len(test_results)
    test_results['avg_best_fitness'] = [avg_best_fitness] * len(test_results)
    test_results['avg_mean_fitness'] = [avg_mean_fitness] * len(test_results)

    if verbose == True:
        print '\n\n' + ('_' * 200)
        print tabulate([['Count of Optimal Runs', 'Percent Runs to Optimal Fitness',
                                  'Avg Best Fitness', 'Avg Mean Fitness'],
                                 [count_opt_runs, opt_run_perc, avg_best_fitness, avg_mean_fitness]],
                                headers='firstrow')


        print '\n\n' + tabulate([['Avg Fitness Evaluations to Convergence', 'Std Dev. Fitness Evaluations',
                                  'Coeff. of Variation'], [avg_evals, std_evals, cv]], headers='firstrow')

        print '\n\n'

    return test_results