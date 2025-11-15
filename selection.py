
# selection.py

import random

# For all the functions here, it's strongly recommended to
# review the documentation for Python's random module:
# https://docs.python.org/3/library/random.html

# Parent selection functions---------------------------------------------------
def uniform_random_selection(population, n, **kwargs):
    select = []
    for _ in range(n):
        select.append(random.choice(population))
    return select


def k_tournament_with_replacement(population, n, k, **kwargs):
    select = []
    for _ in range(n):
        competitors = random.sample(population, k)
        winner = max(competitors, key=lambda individual: individual.fitness)
        select.append(winner)
    return select


def fitness_proportionate_selection(population, n, **kwargs):
    min_fitness = min(individual.fitness for individual in population)
    offset = -min_fitness if min_fitness < 0 else 0
    adjusted_fitnesses = [individual.fitness + offset for individual in population]
    return random.choices(population, weights=adjusted_fitnesses, k=n)



# Survival selection functions-------------------------------------------------
def truncation(population, n, **kwargs):
    sorted_population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
    return sorted_population[:n]


def k_tournament_without_replacement(population, n, k, **kwargs):
    candidates = list(population)
    select = []
    
    for _ in range(n):
        competitor_idc = random.sample(range(len(candidates)), k)
        winner_i = max(competitor_idc, key=lambda i: candidates[i].fitness)
        winner = candidates[winner_i]
        select.append(winner)
        candidates[winner_i] = candidates[-1]
        candidates.pop()
    return select



# Yellow deliverable parent selection function---------------------------------
def stochastic_universal_sampling(population, n, **kwargs):
    min_fitness = min(individual.fitness for individual in population)
    offset = -min_fitness if min_fitness < 0 else 0
    adjusted_fit = [individual.fitness + offset for individual in population]
    total_fitness = sum(adjusted_fit)
    
    #going to use a roulette wheel to determine scholastic spacing
    pointer_spacing = total_fitness / n
    start_pointer = random.uniform(0, pointer_spacing)
    pointers = [start_pointer + i * pointer_spacing for i in range(n)]

    cumulative = 0
    current_i = 0
    select = []
    
    for pointer in pointers:
        #advance along the roulette wheel until the total weight exceeds the pointer
        while current_index < len(population) - 1 and cumulative + adjusted_fit[current_i] < pointer:
            cumulative += adjusted_fit[current_i]
            current_index += 1
        select.append(population[current_i])
    return select