
# gpac_population_evaluation.py

from fitness import *


# 2b TODO: Evaluate the population and assign base_fitness, fitness, and log
#          member variables as described in the Assignment 2b notebook.
def base_population_evaluation(population, parsimony_coefficient, experiment, **kwargs):
    if experiment.casefold() == 'green':
        # We iterate through every individual so that each controller receives a fitness evaluation before selection and survival decisions are made.
        # Evaluating sequentially keeps the logic clear and ensures that per-individual bookkeeping occurs alongside the game simulation.
        for individual in population:
            # We select the object to hand to play_GPac by preferring the individual's parse tree when it exists, which guarantees that the game receives an actual controller rather than a placeholder. Falling back to the individual itself keeps compatibility with helper code that expects TreeGenotype to act as the controller wrapper.
            controller = individual.genes if getattr(individual, 'genes', None) is not None else individual
            # Each evaluation calls play_GPac with the resolved controller, returning both the numeric score and the detailed game log. Passing along **kwargs forwards the configuration parameters such as map and random seeds, preserving the caller's experimental setup.
            score, log = play_GPac(controller, **kwargs)
            # The raw game score becomes the base_fitness because Assignment 2b requires that evolutionary comparisons rely on unpenalized performance. Storing it explicitly allows later analysis to distinguish between true performance and parsimony-adjusted fitness values.
            individual.base_fitness = score
            # Tree size is measured using the node count so that the parsimony objective reflects the overall complexity of the evolved program. We guard against missing genes by treating unevaluated individuals as having zero nodes, which prevents crashes during debugging.
            size = individual.genes.node_count() if getattr(individual, 'genes', None) else 0
            # Multiobjective optimization tracks the original fitness separately from complexity so we intentionally retain the raw score as the single-objective fitness used by legacy utilities. This choice keeps existing selection operators functional while the objectives list carries the richer multiobjective data.
            individual.fitness = score
            # Each individual records the objectives tuple with the first entry counting nodes to be minimized and the second entry recording the score to be maximized. Storing the pair avoids collapsing the objectives into a single penalty value, which supports Pareto-based survival in later pipeline stages.
            individual.objectives = (size, score)
            # We store the game log on the individual so that the highest-performing solutions retain their match history for later analysis and reporting. Retaining the log here also allows the memory-pruning logic after the conditional block to discard unneeded logs safely.
            individual.log = log
        # After evaluating every individual we leave the remaining code to prune excess logs, matching the memory-saving behavior described in the notebook.
        # No additional handling is required here because the shared logic below automatically removes logs from non-elite individuals.

    elif experiment.casefold() == 'yellow':
        # YELLOW: Evaluate a population of Pac-Man controllers against the default ghost agent.
        # Apply parsimony pressure as a second objective to be minimized, rather than a penalty.
        # Sample call: score, log = play_GPac(controller, **kwargs)
        raise NotImplementedError('YELLOW experiment evaluation is not implemented in this helper')

    elif experiment.casefold() == 'red1':
        # RED1: Evaluate a population of Pac-Man controllers against the default ghost agent.
        # Use the score vectors to calculate fitness sharing.
        # Sample call: score, log, score_vector = play_GPac(controller, score_vector=True, **kwargs)
        raise NotImplementedError('RED1 experiment evaluation is not implemented in this helper')

    elif experiment.casefold() == 'red2':
        # RED2: Evaluate a population of Pac-Man controllers against the default ghost agent.
        # Sample call: score, log = play_GPac(controller, **kwargs)
        raise NotImplementedError('RED2 experiment evaluation is not implemented in this helper')

    elif experiment.casefold() == 'red3':
        # RED3: Evaluate a population where each game has multiple different Pac-Man controllers.
        # You must write your own play_GPac_multicontroller function, and use that.
        raise NotImplementedError('RED3 experiment evaluation is not implemented in this helper')

    elif experiment.casefold() == 'red4':
        # RED4: Evaluate a population of ghost controllers against the default Pac-Man agent.
        # Sample call: score, log = play_GPac(None, controller, **kwargs)
        raise NotImplementedError('RED4 experiment evaluation is not implemented in this helper')

    elif experiment.casefold() == 'red5':
        # RED5: Evaluate a population where each game has multiple different ghost controllers.
        # You must write your own play_GPac_multicontroller function, and use that.
        raise NotImplementedError('RED5 experiment evaluation is not implemented in this helper')
    else:
        # Experiments outside the implemented set raise an error so that calling code knows additional logic is required for that configuration.
        # Raising NotImplementedError makes the limitation explicit and avoids silent failures that could compromise experimental results.
        raise NotImplementedError(f'Experiment type {experiment} is not implemented in base_population_evaluation')


    # This code will strip out unnecessary log member variables, to save your memory.
    # We remove the log from any individual that doesn't have a maximal fitness.
    max_fit = max(individual.fitness for individual in population)
    max_base_fit = max(individual.base_fitness for individual in population)
    for individual in population:
        if individual.fitness != max_fit and individual.base_fitness != max_base_fit:
            del individual.log
            individual.log = None

