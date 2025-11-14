
# genetic_programming.py

import random
from base_evolution import BaseEvolutionPopulation

class GeneticProgrammingPopulation(BaseEvolutionPopulation):
    def generate_children(self):
        children = list()
        recombined_child_count = 0
        mutated_child_count = 0

        # We continue producing offspring until the requested quota is met so that each generation supplies enough candidates for evaluation. Using a while loop lets us draw fresh parents on demand depending on whether we mutate or recombine next.
        while len(children) < self.num_children:
            # Sampling a random number per child decides whether this offspring will arise from mutation, using the configured mutation rate as the probability threshold. Including a single-individual safeguard keeps the algorithm functional when the population collapses to one parent during debugging.
            if random.random() < self.mutation_rate or len(self.population) == 1:
                # When mutating we ask the parent selection operator for a single individual so that selection pressure matches the experiment configuration. Requesting exactly one parent here mirrors the interface used throughout the base evolution code.
                parent = self.parent_selection(self.population, 1, **self.parent_selection_kwargs)[0]
                # The mutate method returns a brand-new individual containing a modified copy of the parent's tree, which is the standard subtree mutation operator for GP. Passing the configured keyword arguments preserves depth limits and primitive sets from the experiment file.
                child = parent.mutate(**self.mutation_kwargs)
                # Adding the mutant to the children list immediately ensures it will be evaluated alongside crossover offspring later in the generation. Keeping the append close to the mutation call also simplifies debugging by preserving chronological order.
                children.append(child)
                # Tracking how many offspring came from mutation helps satisfy the notebook's logging requirements and reveals whether the mutation rate behaves as expected. This counter feeds directly into the per-generation log messages below.
                mutated_child_count += 1
            else:
                # For crossover we request two parents at once so that the selection method can apply its pressure consistently to both participants. Drawing them together also prevents reusing a stale partner when the selection method includes randomness or tournaments.
                parent_one, parent_two = self.parent_selection(self.population, 2, **self.parent_selection_kwargs)
                # We call recombine on the first parent while passing the second as the mate so the subtree crossover operator can build a child under the configured depth limits. Using the stored recombination kwargs keeps the behavior aligned with the assignment's specifications.
                child = parent_one.recombine(parent_two, **self.recombination_kwargs)
                # Appending the crossover child to the list guarantees it will be included in the subsequent evaluation and survival phases. Maintaining the same append pattern as the mutation branch keeps the resulting list order meaningful for debugging traces.
                children.append(child)
                # Incrementing the recombination counter lets us report how many children came from crossover, which is important for diagnosing selection pressure and ensuring mutation is not dominating unexpectedly. These counts also provide a sanity check in the experiment logs.
                recombined_child_count += 1


        self.log.append(f'Number of children: {len(children)}')
        self.log.append(f'Number of recombinations: {recombined_child_count}')
        self.log.append(f'Number of mutations: {mutated_child_count}')

        return children
