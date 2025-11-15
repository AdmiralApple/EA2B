
# selection.py

import random

# For all the functions here, it's strongly recommended to
# review the documentation for Python's random module:
# https://docs.python.org/3/library/random.html

# Parent selection functions---------------------------------------------------
def uniform_random_selection(population, n, **kwargs):
    # This line creates an empty list that will store every individual chosen during the sampling process. This structure is necessary because we need to collect exactly n selections while preserving the original order in which they were drawn.
    selected = []
    # This line iterates exactly n times so that we gather the requested number of selections. Using a simple for-loop keeps the logic explicit and avoids assumptions about helper utilities handling corner cases for us.
    for _ in range(n):
        # This line appends a uniformly random individual from the population into the selection list. Employing random.choice provides sampling with replacement, which directly matches the specification for uniform random parent selection.
        selected.append(random.choice(population))
    # This line returns the list of selected individuals to the caller. Returning the constructed list ensures downstream code receives the sampled individuals without any side effects on the original population.
    return selected


def k_tournament_with_replacement(population, n, k, **kwargs):
    # This line initializes a list that will accumulate each tournament winner. Maintaining this list lets us deliver a full roster of selected parents while preserving duplicates when winners repeat across tournaments.
    selected = []
    # This line loops n times to conduct one tournament per required parent. Structuring the tournaments sequentially keeps the implementation simple and matches the expectation that each tournament yields exactly one parent.
    for _ in range(n):
        # This line selects k distinct competitors for the current tournament from the full population. Using random.sample avoids duplicate competitors within a single tournament while still allowing individuals to appear in multiple tournaments.
        competitors = random.sample(population, k)
        # This line determines the competitor with the highest fitness among the sampled individuals. Leveraging max with a key on fitness implements the elitist bias inherent to tournament selection efficiently.
        winner = max(competitors, key=lambda individual: individual.fitness)
        # This line appends the winning individual to the list of selected parents. Appending preserves the order in which winners were decided and allows future tournaments to include the same individual again, satisfying the with-replacement requirement.
        selected.append(winner)
    # This line returns the completed list of tournament winners. Providing the list concludes the selection procedure while keeping the original population untouched.
    return selected


def fitness_proportionate_selection(population, n, **kwargs):
    # This line computes the minimum fitness across the population so we can shift values when negative fitness scores exist. Determining this offset ensures that the weighting scheme remains valid even when some individuals have sub-zero fitness values.
    min_fitness = min(individual.fitness for individual in population)
    # This line calculates how much we need to shift fitness scores to make them all non-negative. Applying this offset allows us to use probability-based sampling functions that expect non-negative weights.
    offset = -min_fitness if min_fitness < 0 else 0
    # This line constructs the list of adjusted fitness values that act as sampling weights. Creating this list explicitly clarifies the mapping between individuals and their effective probabilities.
    adjusted_fitnesses = [individual.fitness + offset for individual in population]
    # This line sums the adjusted fitnesses to obtain the denominator for probability calculations. Knowing the total weight is necessary to detect degenerate cases and to drive roulette-wheel style selection correctly.
    total_fitness = sum(adjusted_fitnesses)
    # This line checks whether all adjusted fitnesses are zero, indicating no meaningful bias. Falling back to uniform random selection in this situation preserves fairness and avoids division-by-zero errors.
    if total_fitness == 0:
        # This line delegates to uniform_random_selection when the fitness landscape is flat. Reusing the earlier helper avoids duplicating logic and guarantees consistent uniform behavior across the codebase.
        return uniform_random_selection(population, n, **kwargs)
    # This line invokes random.choices to sample n individuals proportionally to their adjusted fitness values. Using the standard library’s weighted sampling function provides a concise and well-tested implementation of roulette-wheel selection with replacement.
    return random.choices(population, weights=adjusted_fitnesses, k=n)



# Survival selection functions-------------------------------------------------
def truncation(population, n, **kwargs):
    # This line creates a new list where individuals are sorted from highest to lowest fitness. Sorting in descending order allows us to identify the elite portion of the population efficiently in a single operation.
    sorted_population = sorted(population, key=lambda individual: individual.fitness, reverse=True)
    # This line slices the top n individuals from the sorted list to form the survivor set. Using slicing returns the elite individuals in one step while ensuring no modifications occur to the original population sequence.
    return sorted_population[:n]


def k_tournament_without_replacement(population, n, k, **kwargs):
    # This line copies the population into a mutable list of candidates that we can shrink safely. Working on a shallow copy preserves the input population while allowing efficient removals of chosen survivors.
    candidates = list(population)
    # This line prepares a list to store the winners of each tournament. Maintaining this list ensures we can return all selected individuals after completing the process.
    selected = []
    # This line repeats the tournament process exactly n times to select the required number of survivors. Running the tournaments sequentially enables us to remove winners and prevent duplicates naturally.
    for _ in range(n):
        # This line draws k distinct indices representing competitors for the current tournament. Sampling indices rather than objects allows us to manipulate the candidates list efficiently when removing winners.
        competitor_indices = random.sample(range(len(candidates)), k)
        # This line finds the index of the competitor with the highest fitness among those drawn. Comparing by index keeps the association with the candidates list so we can remove the winner without extra searches.
        winner_index = max(competitor_indices, key=lambda index: candidates[index].fitness)
        # This line retrieves the winning individual using the computed index. Accessing the object directly lets us record the survivor before modifying the candidates list.
        winner = candidates[winner_index]
        # This line appends the winner to the output list of selected survivors. Appending preserves the order in which winners were determined while satisfying the no-duplicate requirement.
        selected.append(winner)
        # This line replaces the winner’s slot with the final candidate to enable an O(1) removal. This swap-then-pop approach avoids the heavy cost of shifting list contents that a naive removal would incur.
        candidates[winner_index] = candidates[-1]
        # This line removes the last element, which is either the original last candidate or the duplicate entry after the swap. Popping from the end completes the efficient removal so that the winner cannot be selected again.
        candidates.pop()
    # This line returns the list of tournament winners who survived. Providing the accumulated survivors finalizes the selection while guaranteeing no duplicates are present.
    return selected



# Yellow deliverable parent selection function---------------------------------
def stochastic_universal_sampling(population, n, **kwargs):
    # Recall that yellow deliverables are required for students in the grad
    # section but bonus for those in the undergrad section.
    # This line determines the minimum fitness value so that we can normalize the weights if negative scores are present. Accounting for negative fitness ensures the sampling procedure remains mathematically valid.
    min_fitness = min(individual.fitness for individual in population)
    # This line computes the offset needed to shift all fitnesses into the non-negative range. Applying this offset allows us to treat the adjusted scores as probabilities without violating stochastic universal sampling requirements.
    offset = -min_fitness if min_fitness < 0 else 0
    # This line builds a list of adjusted fitness values for every individual. Recording the weights in order is essential because stochastic universal sampling relies on the population order when distributing pointers.
    adjusted_fitnesses = [individual.fitness + offset for individual in population]
    # This line calculates the total adjusted fitness, which represents the full length of the roulette wheel. Knowing this total enables us to space pointers evenly across the probability mass.
    total_fitness = sum(adjusted_fitnesses)
    # This line checks for the degenerate case where all adjusted fitnesses are zero. Delegating to uniform random sampling here maintains fairness and avoids division-by-zero issues when computing pointer spacing.
    if total_fitness == 0:
        # This line reuses the uniform random selection method when the fitness landscape lacks variation. Calling the helper keeps behavior consistent across selection operators and reduces duplicated logic.
        return uniform_random_selection(population, n, **kwargs)
    # This line computes the equal spacing between selection pointers on the roulette wheel. Even spacing is the defining characteristic of stochastic universal sampling and helps reduce sampling variance.
    pointer_spacing = total_fitness / n
    # This line chooses a random starting point within the first interval to avoid systematic bias. The random start ensures that every individual receives opportunities proportional to their weight over repeated runs.
    start_pointer = random.uniform(0, pointer_spacing)
    # This line precomputes all pointer positions that will be used to select individuals. Generating the pointers up front simplifies the subsequent traversal of the roulette wheel.
    pointers = [start_pointer + i * pointer_spacing for i in range(n)]
    # This line initializes the cumulative fitness counter used to track progress along the roulette wheel. Maintaining this running sum allows us to determine which individual each pointer lands on.
    cumulative = 0
    # This line sets the index of the current individual under consideration. Tracking the index lets us advance through the population while comparing pointer positions against the cumulative weights.
    current_index = 0
    # This line creates a list that will hold the individuals chosen by the sampling process. Collecting selections in a list ensures we return results in the order implied by the pointer traversal.
    selected = []
    # This line iterates through every pointer to map it onto the roulette wheel. Handling the pointers sequentially guarantees we respect the sorted order without resetting the cumulative sum.
    for pointer in pointers:
        # This line advances along the roulette wheel until the cumulative weight meets or exceeds the pointer. Using a while-loop allows us to skip over individuals with small weights efficiently.
        while current_index < len(population) - 1 and cumulative + adjusted_fitnesses[current_index] < pointer:
            # This line adds the current individual's weight to the cumulative sum before moving forward. Updating the running total keeps the wheel traversal accurate as we pass each individual.
            cumulative += adjusted_fitnesses[current_index]
            # This line moves the index to the next individual so the loop continues scanning forward. Incrementing the index is necessary to progress through the population without repetition.
            current_index += 1
        # This line appends the current individual because the pointer falls within its fitness interval. Selecting the individual here matches the mechanics of stochastic universal sampling precisely.
        selected.append(population[current_index])
    # This line returns the list of individuals chosen by stochastic universal sampling. Delivering the selections completes the operator while maintaining the original population unmodified.
    return selected