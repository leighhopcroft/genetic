from random import randint, random
from operator import add, mul
import numpy as np
from collections import OrderedDict
from tabulate import tabulate
from decimal import Decimal
import pandas as pd
import matplotlib.pyplot as plt

class BaseGenetic(object):

    def __init__(self, params, obj_func, pop_size, retain_percent, mutation_prob, max_gens=np.inf, opt_fitness='unknown'):

        self.params = params
        self.opt_fitness = opt_fitness
        self.max_gens = max_gens
        self.obj_func = obj_func
        self.pop_size = pop_size
        self.retain_percent = retain_percent
        self.mutation_prob = mutation_prob

        self.poss_permutations = reduce(mul, [len(self.params[param]) for param in self.params])
        self.chromosome_len = len(self.params)

        #self.avg_fitnesses, self.min_fitnesses, self.max_fitnesses, self.best_solutions, self.gens_created = self.optimise()
        #self.best_solution = self.best_solutions[-1]

        #if self.gens_created < self.max_gens:
            #self.converged = True
        #else:
            #self.converged = False

    def create_ind(self):
        """
        Create a member of the population.
        """
        ind = []
        for gene_choices in self.params:
            i_max = len(self.params[gene_choices]) - 1
            gene = randint(0, i_max)
            ind.append(gene)
        return ind

    def create_pop(self):
        """
        Create a number of individuals (i.e. a population).
        """
        pop = [self.create_ind() for x in xrange(self.pop_size)]
        return pop

    def calculate_ind_fitness(self, ind):
        fitness = self.obj_func(ind, self.params)
        return fitness

    def evaluate_pop(self, pop):
        """
        Find average, max, min fitness and best solution for a population.
        """
        pop_results = [(self.calculate_ind_fitness(ind), ind) for ind in pop]
        pop_results = sorted(pop_results)
        summed = reduce(add, (self.calculate_ind_fitness(ind) for ind in pop))
        avg_fitness = summed / (len(pop) * 1)
        min_fitness = pop_results[0][0]
        max_fitness = pop_results[-1][0]
        best_solution = pop_results[0][1]
        return avg_fitness, min_fitness, max_fitness, best_solution

    def select_parents(self, pop):
        graded = [(self.calculate_ind_fitness(ind), ind) for ind in pop]
        graded = [x[1] for x in sorted(graded)]
        retain_length = int(len(graded) * self.retain_percent)
        parents = graded[:retain_length]
        return parents

    def crossover(self, parents):
        # crossover parents to create children
        parents_length = len(parents)
        desired_children = self.pop_size - parents_length
        children = []
        while len(children) < desired_children:
            male = randint(0, parents_length - 1)
            female = randint(0, parents_length - 1)
            if male != female:
                male = parents[male]
                female = parents[female]
                cross_point = randint(1, len(male) - 1)
                child = male[:cross_point] + female[cross_point:]
                children.append(child)
        new_pop = parents + children
        return new_pop

    def mutate(self, pop):
        # mutate some individuals in the parents
        mutated_pop = []
        for ind in pop:
            mutated_ind = []
            for pos_to_mutate in xrange(0, len(ind)):
                # For each gene in the individual,
                if self.mutation_prob > random():
                    # If random number between 0 and 1 is less than mutation_prob,
                    # create a random donor individual to inherit a gene from
                    donor_ind = self.create_ind()
                    mutated_gene = donor_ind[pos_to_mutate]
                    # Append the mutated gene to the mutated_ind
                    mutated_ind.append(mutated_gene)

                else:
                    # append the original un-mutated gene to the mutated_ind
                    gene = ind[pos_to_mutate]
                    mutated_ind.append(gene)

            # Once all genes have been mutated (or not depending on probability),
            # append the mutated individual to the mutated_parents list
            mutated_pop.append(mutated_ind)

        return mutated_pop

    def evolve_new_generation(self, pop):
        # Select Parents
        parents = self.select_parents(pop)

        # Crossover parents to create children and new pop
        new_pop = self.crossover(parents)

        # Mutate new population according to probability
        mutated_pop = self.mutate(new_pop)

        # Grade new population to determine average fitness, min fitness, max fitness and best solution
        avg_fitness, min_fitness, max_fitness, best_solution = self.evaluate_pop(mutated_pop)

        return avg_fitness, min_fitness, max_fitness, best_solution, mutated_pop

    def optimise(self, verbose=True):
        # Create initial population
        pop = self.create_pop()

        # Evaluate the initial population
        init_avg_fitness, init_min_fitness, init_max_fitness, init_best_solution = self.evaluate_pop(pop)

        # Create a running list of all solutions evaluated so far
        solutions = pop

        # Create lists of initial results to append to later
        avg_fitnesses = [init_avg_fitness]
        min_fitnesses = [init_min_fitness]
        max_fitnesses = [init_max_fitness]
        best_solutions = [init_best_solution]

        # Create a counter for generations elapsed, 0 being the initial population
        gen_count = 0

        # Set convergence boolean to False initially
        convergence = False

        while gen_count < self.max_gens and convergence == False:
            # Increment generation counter
            gen_count += 1

            # Evolve and evaluate a new generation
            avg_fitness, min_fitness, max_fitness, best_solution, pop = self.evolve_new_generation(pop)

            solutions += pop

            # Append results to the previously created lists
            avg_fitnesses.append(avg_fitness)
            min_fitnesses.append(min_fitness)
            max_fitnesses.append(max_fitness)
            best_solutions.append(best_solution)

            if gen_count % 500 == 0:
                print('\tGen {:d}'.format(gen_count))

            # Check for convergence
            if self.opt_fitness != 'unknown':
                # If we know the optimal solution (or target) then check for that
                if self.opt_fitness in min_fitnesses:
                    print('Optimisation converged to known optimal solution in {0:,} generations '
                          '({1:,} fitness function evaluations)\n\n'.format(gen_count, gen_count * self.pop_size))
                    convergence = True

            else:
                # If we don't know the optimal solution, then we can only assume convergence
                # when the best solution has remained stable for some time
                # (within max_conv_std standard deviations of conv_lookback amount of prev solutions)
                #max_conv_std = 0.001
                conv_lookback = 10
                if len(min_fitnesses) >= conv_lookback:
                    x = min_fitnesses[-conv_lookback:]  # set x to be the last conv_lookback elems of best_solutions
                    #if np.std(x) <= max_conv_std:
                    if x.count(x[0]) == len(x) == True:  # check if all elements in x are identical
                        print('Optimisation converged to stable solution in {0:,} generations '
                              '({1:,} fitness function evaluations)\nThis was due to no greater change in standard '
                              'deviation of optimal solution than {2:2.3f} in the past {3:d} generations\n\n'.format(
                            gen_count,
                            gen_count * self.pop_size,
                            max_conv_std,
                            conv_lookback
                        ))
                        convergence = True

        if convergence == False:
            # We must have reached the max_gens to break from the while loop above
            print('\n\nMax generations reached\nOptimisation did not converge to optimal/stable/target solution '
                  'in {0:,} generations ({1:,} fitness function evaluations)\n\n'.format(
                gen_count,
                gen_count * self.pop_size,
            ))

        unique_solutions = [list(ind) for ind in set(tuple(ind) for ind in solutions)]

        total_solutions_explored = gen_count * self.pop_size

        if verbose == True:
            print('Unique Solutions Evaluated = {use:,}\n'
                  'Total Solutions Evaluated = {tse:,}\n'
                  'Total Possible Solutions = {tps:,}\n'
                  'Search Space Explored = {pss:3.5f}%\n'
                  'Solutions evaluated more than once = {dc:,}\n'
                  'Efficiency of Computational Resources = {ef:2.2f}%\n\n'.format(
                use=len(unique_solutions),
                tse=total_solutions_explored,
                tps=self.poss_permutations,
                pss=(len(unique_solutions)/float(self.poss_permutations))*100,
                dc=total_solutions_explored-len(unique_solutions),
                ef=(1-(total_solutions_explored-len(unique_solutions))/float(total_solutions_explored))*100
            ))

            print('Final Avg Fitness = {0:3.5f}\n'
                  'Final Best Fitness = {1:3.5f}\n'
                  'Final Best Solution = {2:s}'.format(avg_fitness, min_fitness, best_solution))

            self.decode_ind(best_solution, verbose=True)
            print '\n\n'

        # Once all generations are completed or convergence is achieved, return the results as lists
        # and the gen_count as a value
        return avg_fitnesses, min_fitnesses, max_fitnesses, best_solutions, gen_count

    def decode_ind(self, ind, verbose=False):
        decoded_ind = []
        param_headers = []
        chromosome_idx = 0
        for key in self.params:
            decoded_gene = self.params[key][ind[chromosome_idx]]
            decoded_ind.append(decoded_gene)
            param_headers.append(key)
            chromosome_idx += 1

        param_headers.append('Fitness')
        decoded_ind.append(self.calculate_ind_fitness(ind))

        if verbose == True:
            print '\n\n' + tabulate([param_headers, decoded_ind],headers='firstrow')

        return decoded_ind