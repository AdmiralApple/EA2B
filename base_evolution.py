
# base_evolution.py

import statistics
import random


#calculates domination
def dominates(candidate, challenger):
    
    candidate_nodes, candidate_score = candidate.objectives
    challenger_nodes, challenger_score = challenger.objectives

    no_worse = candidate_nodes <= challenger_nodes and candidate_score >= challenger_score
    strictly_better = candidate_nodes < challenger_nodes or candidate_score > challenger_score
    
    return no_worse and strictly_better


def compute_pareto_front(population):
    # The Pareto front collects individuals that are not dominated by any peers, providing the survival operator with a multiobjective elite set. Iterating over the population ensures every controller receives a fair comparison, even in heterogeneous populations.
    front = []
    # Evaluating domination per individual requires nested comparisons, which remain tractable for the small population sizes used in the assignments. Using a simple list keeps the data structure minimal and avoids premature optimization.
    for individual in population:
        # We assume individuals carry an objectives tuple; if they do not, the AttributeError will surface quickly, revealing evaluation issues early. This assumption matches the updated evaluation pipeline that stores both node count and score on each individual.
        dominated = any(dominates(other, individual) for other in population if other is not individual)
        # Only non-dominated individuals join the front, guaranteeing that the list exclusively contains Pareto-optimal candidates. Appending inside the conditional keeps ordering stable, which is useful for reproducible logging.
        if not dominated:
            front.append(individual)
    # Returning the resulting list provides downstream components with direct access to the elite individuals. The caller can further process or sort the list depending on the survival strategy in use.
    return front


def compute_hypervolume(pareto_front, population):
    # Hypervolume summarizes the quality of the Pareto front by measuring the dominated portion of the objective space. Providing the full population allows us to derive a reference point that is guaranteed to be dominated by every Pareto-optimal individual.
    if not pareto_front:
        # An empty front produces zero hypervolume, reflecting the absence of dominating solutions. Returning early avoids division-by-zero issues in later calculations.
        return 0.0
    # We gather all node counts from the population to identify the largest, which becomes the worst-case reference for the minimizing objective. Using the entire population ensures the reference point is pessimistic enough to remain dominated by the front.
    all_nodes = [individual.objectives[0] for individual in population]
    # Similarly, we collect all scores to locate the smallest observed value as the worst case for the maximizing objective. The minimum score guarantees the reference lies below every Pareto solution in the score dimension.
    all_scores = [individual.objectives[1] for individual in population]
    # The reference node count equals the population maximum, ensuring every Pareto member uses the same or fewer nodes and therefore dominates the reference on the minimizing axis. Adding no extra margin keeps the computation simple while remaining correct for discrete node counts.
    reference_nodes = max(all_nodes)
    # The reference score equals the population minimum so that even the weakest Pareto member scores at least as high. Picking from the same generation maintains consistency with Assignment 1d guidance that hypervolume be relative to observed extremes.
    reference_score = min(all_scores)
    # We convert the Pareto front into maximization coordinates by flipping the sign of node counts, transforming the minimization objective into an equivalent maximization problem. Sorting by score descending stabilizes the integration order needed for a two-dimensional hypervolume sweep.
    points = sorted(((individual.objectives[1], -individual.objectives[0]) for individual in pareto_front), reverse=True)
    # Converting the reference point into the same coordinate system lets us reuse a unified sweep algorithm. The resulting tuple is dominated by every Pareto point because it has the lowest score and the smallest negated node count.
    reference_point = (reference_score, -reference_nodes)
    # We integrate the area under the stepwise Pareto frontier by accumulating rectangular slices. Starting from the reference score ensures that each slice measures the additional dominated volume contributed by the current point.
    hypervolume = 0.0
    # The running score tracks the last slice boundary along the score axis, preventing over-counting when multiple Pareto members share similar scores. Initializing to the reference score keeps the first slice anchored at the dominated region's origin.
    last_score = reference_point[0]
    # We iterate through the sorted points to compute each slice's width along the score dimension and height along the transformed node dimension. This loop mirrors the assignment's geometric visualization and produces an interpretable scalar summary.
    for score, neg_nodes in points:
        # Width equals the improvement in score over the previous slice, reflecting the additional range dominated in the maximizing objective. Clamping negative widths to zero guards against floating-point noise that could otherwise produce a negative contribution.
        width = max(0.0, score - last_score)
        # Height equals the difference between the current negated node value and the reference, which corresponds to reduced complexity. Again we clamp to zero so that small rounding errors never yield negative hypervolume contributions.
        height = max(0.0, neg_nodes - reference_point[1])
        # Each rectangle's area multiplies width and height, matching the textbook definition of two-dimensional hypervolume. Summing the areas across all Pareto members yields the total dominated volume.
        hypervolume += width * height
        # Updating the running score boundary prepares the integration for the next Pareto point. This ensures that successive rectangles abut rather than overlap, keeping the total accurate.
        last_score = score
    # Returning the final hypervolume allows logging and analytics to monitor convergence toward broader Pareto fronts. The value is always non-negative thanks to the earlier clamps.
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


    def survival(self):
        # Recording the population size before survival helps diagnose whether selection pressure is accidentally shrinking or expanding the population over time. This transparency is essential when debugging complex multiobjective behaviors.
        self.log.append(f'Pre-survival population size: {len(self.population)}')
        # We allow the survival operator to return either a bare list of survivors or an enriched tuple containing additional analytics such as the Pareto front and hypervolume. Capturing the raw return value first keeps the method flexible while supporting multiobjective workflows.
        survival_result = self.survival_selection(self.population, self.mu, **self.survival_selection_kwargs)
        # When the survival operator returns a tuple, we unpack the survivors alongside the Pareto diagnostics so that downstream logging reflects the operator's detailed computations. This branch mirrors the Assignment 1d expectation that survival may produce a Pareto front directly.
        if isinstance(survival_result, tuple):
            survivors, pareto_front, hypervolume = survival_result
            # Unpacking into named variables keeps subsequent logic readable by clarifying which value stores survivors versus analytics. This structure also simplifies future extensions if additional statistics are returned alongside the tuple.
        else:
            # If only survivors are returned we compute the Pareto front ourselves to maintain consistent analytics. This fallback guarantees that even basic survival operators benefit from the new logging without code duplication.
            survivors = survival_result
            # Storing the survivors explicitly preserves compatibility with the tuple path and ensures the later assignment always references a well-defined list. This line also offers a convenient breakpoint when debugging survival outputs.
            pareto_front = compute_pareto_front(survivors)
            # Computing the Pareto front here guarantees that downstream logging always reflects non-dominated solutions even when the survival operator does not supply them. Performing the calculation locally avoids forcing every survival strategy to implement duplicate logic.
            hypervolume = compute_hypervolume(pareto_front, survivors)
            # Hypervolume is derived immediately after the front so that both statistics describe the same survivor set. Calculating it in this branch keeps the reporting symmetric with the tuple path where the value might be precomputed.
        # The population is updated after extracting diagnostics to keep the attribute aligned with the actual survivors list. Making the assignment explicit also clarifies when the new generation replaces the old.
        self.population = survivors
        # Logging the post-survival size verifies that exactly mu individuals remain and provides a sanity check when experimenting with alternative survival operators. Immediate visibility into mismatched counts can prevent subtle bugs from propagating.
        self.log.append(f'Post-survival population size: {len(self.population)}')
        # We append multiobjective statistics only when every survivor provides the required objectives to avoid raising exceptions in legacy single-objective workflows. This conditional keeps the new analytics informative without sacrificing backward compatibility.
        if all(hasattr(individual, 'objectives') for individual in self.population):
            # When objectives are present we forward the previously computed Pareto data into the logging helper for detailed transparency. This mirrors Assignment 1d expectations about reporting fronts and hypervolume each generation.
            self.log_multiobjective_stats(pareto_front, hypervolume)
        else:
            # If objectives are missing we still record a clear message so experiment logs document why multiobjective metrics were skipped. This helps practitioners identify configurations that need updated evaluation code.
            self.log.append('Multiobjective statistics unavailable: missing objectives')


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
        # Logging starts by ensuring we have a Pareto front and hypervolume, computing them on demand to keep callers lightweight. This design allows both survival() and external utilities to reuse the method without redundant calculations.
        if pareto_front is None:
            # When no Pareto front is provided we derive it from the current population to keep metrics aligned with the latest survivors. This ensures consistent reporting even if survival() bypasses the helper and updates self.population directly.
            pareto_front = compute_pareto_front(self.population)
        if hypervolume is None:
            # Hypervolume defaults to an on-the-fly calculation using the same Pareto front so that logging never omits this summary. Performing the computation here avoids forcing every caller to repeat the logic.
            hypervolume = compute_hypervolume(pareto_front, self.population)
        # Capturing the current evaluation count maintains continuity with other logging helpers and shows how progress correlates with Pareto improvements. This context is invaluable when visualizing convergence across runs.
        self.log.append(f'Evaluations: {self.evaluations}')
        # We precompute lists of nodes and scores to reuse for both min and mean statistics, which avoids recomputing the same generators multiple times. Materializing the values also simplifies debugging because the numbers can be inspected directly.
        population_nodes = [individual.objectives[0] for individual in self.population]
        population_scores = [individual.objectives[1] for individual in self.population]
        # The best node count reports the minimum complexity present in the full population, emphasizing the minimization objective. Using min instead of max aligns the statistic with the parsimony goal introduced by the second objective.
        self.log.append(f'Local best node count: {min(population_nodes)}')
        # The mean node count conveys how the population balances simplicity on average, offering a holistic view beyond the single best individual. Monitoring this value helps identify premature bloat or over-penalization.
        self.log.append(f'Local mean node count: {statistics.mean(population_nodes)}')
        # The best score highlights the maximum raw performance still present after survival, reinforcing the maximization objective. This value mirrors traditional single-objective logging for compatibility with earlier analyses.
        self.log.append(f'Local best score: {max(population_scores)}')
        # The mean score offers insight into overall competency across the population, making it easier to gauge whether the algorithm is improving consistently. Pairing this with the node metrics demonstrates trade-offs between performance and complexity.
        self.log.append(f'Local mean score: {statistics.mean(population_scores)}')
        # Reporting the Pareto front size reveals how many diverse solutions survived, which is a useful health indicator for multiobjective evolution. A shrinking front may signal overly aggressive selection pressure.
        self.log.append(f'Individuals in the Pareto front: {len(pareto_front)}')
        # We guard against an empty Pareto front by supplying default zero values, ensuring the logger never raises an exception when survival returns no elite individuals. Even though this scenario should be rare, the safeguard keeps experiments running smoothly.
        if pareto_front:
            # When the front has members we collect their nodes and scores to reuse for both statistics, mirroring the approach used for the full population. This pattern keeps the code consistent and readable.
            front_nodes = [individual.objectives[0] for individual in pareto_front]
            front_scores = [individual.objectives[1] for individual in pareto_front]
            # The front's mean node count summarizes complexity among elite solutions, helping practitioners decide whether additional parsimony pressure is necessary. Keeping this metric alongside the population-wide numbers exposes selection effects directly.
            self.log.append(f'Pareto front mean node count: {statistics.mean(front_nodes)}')
            # The front's mean score complements the node metric and demonstrates how well the elite set performs relative to the broader population. Analysts can track this statistic to ensure that simplification does not come at the cost of capability.
            self.log.append(f'Pareto front mean score: {statistics.mean(front_scores)}')
        else:
            # If the front is empty we fall back to zeros so that downstream consumers still receive numeric data. Including the entry maintains consistent log formatting across generations.
            self.log.append('Pareto front mean node count: 0.0')
            self.log.append('Pareto front mean score: 0.0')
        # Finally, we record the hypervolume to provide a single scalar capturing the Pareto front's coverage of objective space. Including this value in the log offers a convenient progress indicator for Assignment 1d experiments.
        self.log.append(f'Pareto front hypervolume: {hypervolume}')
