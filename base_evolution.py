
# base_evolution.py

import statistics
import random


#calculates whether the candidate dominates the challenger
def dominates(candidate, challenger):
    
    candidate_nodes, candidate_score = candidate.objectives
    challenger_nodes, challenger_score = challenger.objectives

    no_worse = candidate_nodes <= challenger_nodes and candidate_score >= challenger_score
    strictly_better = candidate_nodes < challenger_nodes or candidate_score > challenger_score
    
    return no_worse and strictly_better


def compute_pareto_front(population):
    pareto_front = []

    for individual in population:
        dominated = False

        for other in population:
            if dominates(other, individual):
                dominated = True
                break

        #if not dominated, add it to the pareto front
        if not dominated:
            pareto_front.append(individual)

    return pareto_front


#measure the dominated portion of the objective space
def compute_hypervolume(pareto_front, population):
    nodes = [individual.objectives[0] for individual in population]
    node_max = max(nodes)

    scores = [individual.objectives[1] for individual in population]
    score_min = min(scores)

    #flip the coordinates so that both objectives are maximized
    points = sorted(((individual.objectives[1], -individual.objectives[0]) for individual in pareto_front), reverse=True)
    reference = (score_min, -node_max)
    
    hypervolume = 0.0
    last_score = reference[0]
    
    for score, neg_nodes in points:
        width       = max(0.0, score - last_score)
        height      = max(0.0, neg_nodes - reference[1])
        hypervolume += width * height
        last_score  = score
    return hypervolume

class BaseEvolutionPopulation():
    def __init__(self, individual_class, mu, num_children,
                 mutation_rate, parent_selection, survival_selection,
                 problem=dict(), parent_selection_kwargs=dict(),
                 recombination_kwargs=dict(), mutation_kwargs=dict(),
                 survival_selection_kwargs=dict(), **kwargs):
        self.mu = mu
        self.num_children = num_children
        self.mutation_rate = mutation_rate
        self.parent_selection = parent_selection
        self.survival_selection = survival_selection
        self.parent_selection_kwargs = parent_selection_kwargs
        self.recombination_kwargs = recombination_kwargs
        self.mutation_kwargs = mutation_kwargs
        self.survival_selection_kwargs = survival_selection_kwargs

        self.log = []
        self.log.append(f'mu: {self.mu}')
        self.log.append(f'num_children: {self.num_children}')
        self.log.append(f'mutation rate: {self.mutation_rate}')
        self.log.append(f'parent selection: {self.parent_selection.__name__ }')
        self.log.append(f'parent selection kwargs: {self.parent_selection_kwargs}')
        self.log.append(f'survival selection: {self.survival_selection.__name__ }')
        self.log.append(f'survival selection kwargs: {self.survival_selection_kwargs}')
        self.log.append(f'recombination kwargs: {self.recombination_kwargs}')
        self.log.append(f'mutation kwargs: {self.mutation_kwargs}')

        self.population = individual_class.initialization(self.mu, **problem, **kwargs)
        self.evaluations = 0

        self.log.append(f'Initial population size: {len(self.population)}')


    def generate_children(self):
        # Randomly select self.num_children * 2 parents using your selection algorithm
        parents = self.parent_selection(self.population, self.num_children * 2, **self.parent_selection_kwargs)
        random.shuffle(parents)

        children = list()
        mutated_child_count = 0

        # TODO: Get pairs of parents
        # HINT: range(0, len(parents), 2) to iterate two at a time

        # TODO: Recombine each pair to generate a child
        # HINT: With parents p1 and p2, get a child with
        #       p1.recombine(p2, **self.recombination_kwargs)


        # TODO: Mutate each child independently with probability self.mutation_rate.
        #       The probability is independent for each child, meaning you should
        #       randomly decide if each individual child gets mutated. That is,
        #       you shouldn't calculate a precise number of mutations to occur ahead of time.
        #       Record the number of mutated children in mutated_child_count variable.
        # HINT: With a recombined child, get a mutated copy with
        #       child.mutate(**self.mutation_kwargs)
        #       Keep in mind this does not modify child, it returns a new object

        self.log.append(f'Number of children: {len(children)}')
        self.log.append(f'Number of mutations: {mutated_child_count}')

        return children

    #run one survival step
    def survival(self):
        self.log.append(f'Pre-survival population size: {len(self.population)}')
        survivors = self.survival_selection(self.population, self.mu, **self.survival_selection_kwargs)

        pareto_front = compute_pareto_front(survivors)
        hypervolume = compute_hypervolume(pareto_front, survivors)

        #replace the old population with the survivors
        self.population = survivors

        self.log.append(f'Post-survival population size: {len(self.population)}')
        self.log_multiobjective_stats(pareto_front, hypervolume)



    def log_base_stats(self):
        self.log.append(f'Evaluations: {self.evaluations}')
        self.log.append(f'Local best: {max(map(lambda x:x.fitness, self.population))}')
        self.log.append(f'Local mean: {statistics.mean(map(lambda x:x.fitness, self.population))}')


    def log_penalized_stats(self):
        self.log.append(f'Evaluations: {self.evaluations}')
        self.log.append(f'Local best penalized fitness: {max(map(lambda x:x.fitness, self.population))}')
        self.log.append(f'Local mean penalized fitness: {statistics.mean(map(lambda x:x.fitness, self.population))}')
        self.log.append(f'Local best base fitness: {max(map(lambda x:x.base_fitness, self.population))}')
        self.log.append(f'Local mean base fitness: {statistics.mean(map(lambda x:x.base_fitness, self.population))}')
        self.log.append(f'Number of valid solutions: {[x.violations for x in self.population].count(0)}')


    def log_multiobjective_stats(self, pareto_front=None, hypervolume=None):
        
        self.log.append(f'Evaluations: {self.evaluations}')

        #precompute lists of nodes and scores to reuse
        population_nodes = [individual.objectives[0] for individual in self.population]
        population_scores = [individual.objectives[1] for individual in self.population]

        self.log.append(f'Local best node count: {min(population_nodes)}')
        self.log.append(f'Local mean node count: {statistics.mean(population_nodes)}')
        self.log.append(f'Local best score: {max(population_scores)}')
        self.log.append(f'Local mean score: {statistics.mean(population_scores)}')
        self.log.append(f'Individuals in the Pareto front: {len(pareto_front)}')
        
        #define these here because it looks cleaner
        front_nodes = [individual.objectives[0] for individual in pareto_front]
        front_scores = [individual.objectives[1] for individual in pareto_front]

        self.log.append(f'Pareto front mean node count: {statistics.mean(front_nodes)}')
        self.log.append(f'Pareto front mean score: {statistics.mean(front_scores)}')
        self.log.append(f'Pareto front hypervolume: {hypervolume}')
