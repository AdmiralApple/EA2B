# tree_genotype.py

# The random module provides all stochastic behavior required for sampling primitives and choosing crossover points, so it must be imported explicitly.
# I rely on the standard library implementation because the assignment framework already uses it extensively, which keeps reproducibility consistent across modules.
import random
# The deepcopy function duplicates nested tree structures without sharing references, making it indispensable for safe recombination and mutation.
# Using deepcopy from the standard copy module prevents subtle bugs that would arise if mutated children altered their parents unintentionally.
from copy import deepcopy
# The math module supplies infinity, which we use to represent unreachable distances cleanly when a particular sensor has no valid targets.
# Importing inf directly keeps the evaluation logic succinct while still leveraging Python's robust floating-point handling.
from math import inf
# The manhattan helper is already implemented in the fitness module, so reusing it avoids duplicating distance calculations and keeps the code aligned with the assignment instructions.
# Importing this function also ensures that any future adjustments to distance calculations in the fitness module automatically propagate to the tree evaluation logic.
from fitness import manhattan

# The following constant defines the inclusive range for sampled constant terminals so that every constant falls within a reasonable magnitude compared to the other sensors.
# I chose a symmetric range around zero because it lets the tree offset or amplify sensor values in either direction without dominating the arithmetic operations.
CONSTANT_RANGE = (-10.0, 10.0)

# This epsilon establishes the tolerance used to detect potentially unsafe denominators during protected division.
# I selected a very small value so that the implementation only treats near-zero denominators as dangerous while allowing legitimate small values to behave normally.
DIVISION_EPSILON = 1e-9

# The TreeNode class models individual primitives so that evaluation, serialization, and structural edits can all interact with a common representation.
# Storing nodes as objects rather than nested tuples keeps the implementation readable and makes in-place subtree replacement straightforward during crossover and mutation.
class TreeNode:
    # The initializer records the primitive label, an optional constant value, and a list of child references so that each node encapsulates everything needed for evaluation.
    # I prefer storing children in a list because lists make it easy to swap out subtrees without reconstructing the entire parent node, which keeps evolutionary operations efficient.
    def __init__(self, primitive, value=None, children=None):
        # The primitive string identifies the sensor or operator represented by this node, so we keep it unmodified from the configuration to simplify serialization later.
        # Using the raw string also avoids additional mapping layers that could introduce mistakes when loading and saving trees.
        self.primitive = primitive
        # Constant nodes require a numeric payload to remain stable across evaluations, while non-constant primitives simply leave this attribute as None.
        # Storing the value directly on the node keeps the evaluation logic fast because it does not need to consult any external tables.
        self.value = value
        # Children are stored as a list so that other algorithms can iterate and replace entries easily, and we copy incoming iterables to avoid unexpected shared references.
        # Using an empty list for terminals highlights their status as leaves, which allows helper methods to detect terminal nodes without relying on primitive names alone.
        self.children = list(children) if children else []

    # Terminal detection is based on the presence of children because that criterion remains accurate regardless of the primitive naming scheme.
    # Returning True when the children list is empty keeps the logic concise and avoids storing redundant boolean flags on every node.
    def is_terminal(self):
        # Using the not operator on the children list leverages Python's truthiness rules so that empty lists evaluate to False while populated lists evaluate to True.
        # This idiom keeps the method compact and naturally updates whenever the child list changes.
        return not self.children

    # Evaluation combines the results from child nodes recursively, using optional caching to avoid recomputing expensive sensor values.
    # Accepting a cache parameter enables the caller to reuse the same dictionary across the entire tree evaluation, which substantially reduces redundant work for large trees.
    def evaluate(self, state, cache=None):
        # We ensure that a mutable cache exists before using it so that callers who are unconcerned with caching can pass None without triggering errors.
        # Creating the dictionary lazily also avoids unnecessary allocations in scenarios where the tree contains no duplicate sensors.
        cache = cache if cache is not None else dict()
        # Constant nodes return their intrinsic value immediately because they do not depend on the game state.
        # Handling this case early keeps the remainder of the method focused on sensors and operators, reducing branching in the hot path.
        if self.primitive == 'C':
            # Returning the stored value directly maintains determinism for constant terminals and avoids polluting the cache with redundant entries.
            # This early exit also keeps the method efficient when trees contain many constants, which is a common occurrence after mutation.
            return self.value
        # Sensors are identified by their lack of children, and we compute their values using helper methods while caching results to avoid duplicate calculations.
        # This approach preserves flexibility because adding a new sensor only requires updating the helper without touching the evaluation loop.
        if self.is_terminal():
            # Using the primitive name as the cache key ensures that identical sensors share cached values within a single evaluation, which aligns with the assignment's performance advice.
            # This strategy also keeps the cache small and easy to inspect during debugging sessions.
            key = self.primitive
            # If the cache already holds a value for this sensor we can return it immediately, thereby skipping another potentially expensive distance computation.
            # Leveraging cached values can dramatically reduce runtime when trees reuse the same sensor many times, as often happens after recombination.
            if key in cache:
                # Returning the cached value provides the caller with the exact same result produced earlier in the evaluation, ensuring correctness while saving computation time.
                # This branch also keeps the recursion shallow because no further helper calls are required for cached values.
                return cache[key]
            # When no cached value exists we compute the sensor reading from the state using a dedicated helper function.
            # Offloading the actual calculation keeps this method focused on orchestration while the helper manages the details for each primitive.
            value = self._evaluate_terminal(state)
            # Once computed we store the value under the primitive key so that any additional occurrences of the same sensor can reuse it instantly.
            # Caching here adheres to the assignment's guidance and significantly improves evaluation speed for deep trees.
            cache[key] = value
            # Returning the freshly computed value allows parent nodes to incorporate the sensor output into their calculations without waiting for another recursion layer.
            # The raw float is returned without rounding to preserve as much information as possible for the evolutionary process.
            return value
        # Operator nodes require evaluating both children first so that their results can be combined according to the primitive's semantics.
        # We evaluate the left child before the right child to maintain deterministic behavior, which is particularly important for non-commutative operators like subtraction.
        left = self.children[0].evaluate(state, cache)
        # The right child is evaluated using the same cache so that any terminals shared between subtrees still benefit from memoization.
        # Storing the result in a local variable ensures it will be reused without another recursive call during the operator dispatch below.
        right = self.children[1].evaluate(state, cache)
        # Addition returns the sum of both operands, matching the specification exactly and providing a simple linear combination tool for the GP.
        # I implement the behavior inline rather than referencing a lookup table to keep the control flow explicit and easy to debug.
        if self.primitive == '+':
            # Returning the computed sum directly respects the intended semantics and keeps the implementation fast.
            # No additional safeguards are needed here because floating-point addition handles all numeric inputs gracefully.
            return left + right
        # Subtraction maintains operand order so that the tree can express directional relationships between sensors.
        # Keeping this branch separate from addition avoids subtle mistakes where operand ordering might be reversed inadvertently.
        if self.primitive == '-':
            # We return the difference between left and right to mirror the mathematical operator precisely.
            # This design supports strategies where Pac-Man prefers or avoids certain features relative to others.
            return left - right
        # Multiplication combines operands multiplicatively, enabling GP to scale sensor values or construct interaction terms.
        # Python's floating-point multiplication is reliable for the magnitudes produced by our sensors, so no additional guarding is necessary here.
        if self.primitive == '*':
            # Returning the product makes the operator available to the evolutionary process without additional wrapping logic.
            # This branch remains simple by design to keep evaluation overhead minimal.
            return left * right
        # Protected division guards against near-zero denominators so that the tree never generates NaNs or infinities during evaluation.
        # I compare the absolute denominator to the pre-defined epsilon because this approach is both fast and numerically stable.
        if self.primitive == '/':
            # When the denominator is dangerously small we fall back to returning the numerator, which is a common protected-division strategy in GP literature.
            # This choice keeps the output finite and preserves continuity, which helps the evolutionary search remain stable.
            if abs(right) <= DIVISION_EPSILON:
                # Returning the numerator in the protected case avoids undefined behavior while still allowing the expression to propagate meaningful values.
                # This fallback also mirrors the assignment requirement to handle division by zero gracefully.
                return left
            # If the denominator is sufficiently large we perform standard floating-point division.
            # Returning the quotient directly gives the tree access to rational relationships, which can be useful for balancing multiple sensor signals.
            return left / right
        # The RAND operator samples a uniform random value between the two operands so that the controller can inject stochastic behavior into its policy.
        # Sorting the operands first ensures that the sampled interval is valid regardless of the order in which the children evaluate.
        if self.primitive == 'RAND':
            # Using sorted returns the operands in ascending order, giving us the correct lower and upper bounds for the random sampler.
            # This technique handles all numeric inputs gracefully, even when both operands are equal.
            low, high = sorted((left, right))
            # random.uniform performs the sampling efficiently and integrates seamlessly with the rest of the assignment's stochastic components.
            # Returning the sampled value allows the operator to contribute randomness exactly as described in the assignment documentation.
            return random.uniform(low, high)
        # Any primitive that reaches this point is unknown, so we raise a ValueError to alert the developer immediately.
        # Failing fast here prevents silent logic errors that would be much harder to diagnose later in the evolutionary process.
        raise ValueError(f'Unsupported primitive: {self.primitive}')

    # Terminal evaluation requires inspecting the game state, so we encapsulate that behavior inside a dedicated helper method.
    # Centralizing this logic keeps evaluate concise and makes it straightforward to add or modify sensor primitives in the future.
    def _evaluate_terminal(self, state):
        # Pac-Man's position is needed by multiple sensors, so we compute it once using a helper that correctly handles both single and multi-Pac scenarios.
        # Delegating to a helper also ensures consistency with any future modules that need to interpret the players dictionary.
        pac_location = find_pac_location(state['players'])
        # Ghost positions are collected through another helper because only keys without the letter 'm' correspond to ghosts in the provided naming scheme.
        # Caching the result locally avoids scanning the dictionary repeatedly inside each sensor branch.
        ghost_locations = find_ghost_locations(state['players'])
        # The ghost-distance sensor returns the Manhattan distance to the nearest ghost, defaulting to infinity when no ghosts are present.
        # Returning infinity in the empty case avoids crashes and clearly indicates that ghosts are unreachable from the current state.
        if self.primitive == 'G':
            return min((manhattan(pac_location, ghost) for ghost in ghost_locations), default=inf)
        # The pill sensor finds the closest pill using the same Manhattan distance metric and also defaults to infinity when no pills remain.
        # This behavior mirrors the ghost sensor so that downstream arithmetic can treat both sensors consistently.
        if self.primitive == 'P':
            return min((manhattan(pac_location, pill) for pill in state['pills']), default=inf)
        # The fruit sensor returns the Manhattan distance when a fruit exists; otherwise it returns zero so that fruit-less states share a constant value as required.
        # Using zero keeps the value finite and neutral, aligning with the assignment's instruction that all fruit-free states should behave identically.
        if self.primitive == 'F':
            return manhattan(pac_location, state['fruit']) if state['fruit'] is not None else 0.0
        # The wall sensor counts how many of the four cardinal directions are blocked, so we call a helper that already accounts for map boundaries.
        # This helper encapsulates the wall logic used elsewhere, ensuring that the W sensor remains consistent with the environment representation.
        if self.primitive == 'W':
            return count_adjacent_walls(pac_location, state['walls'])
        # Any other primitive reaching this point indicates a mismatch between the configuration and the evaluation logic, so we raise an error immediately.
        # Explicitly failing here helps surface configuration mistakes during development rather than letting them propagate silently.
        raise ValueError(f'Unsupported terminal primitive: {self.primitive}')

    # The height method computes the maximum depth of the subtree rooted at this node so that the algorithm can enforce depth limits during genetic operations.
    # Returning the depth as an integer keeps the logic compatible with the depth limit specified in the configuration files.
    def height(self):
        # Terminal nodes contribute a height of one because they occupy a single level in the tree.
        # Returning early in this case avoids unnecessary recursion and clarifies that leaves always satisfy the depth constraint individually.
        if self.is_terminal():
            return 1
        # Internal nodes return one plus the maximum child height, which mirrors the standard definition of tree height and ensures accuracy for unbalanced trees.
        # Using a generator expression keeps the code concise while avoiding intermediate list allocations for the child heights.
        return 1 + max(child.height() for child in self.children)

    # The node_count method tallies the number of primitives in the subtree, providing the metric used for parsimony pressure.
    # Counting nodes rather than depth captures both wide and deep complexity, which better reflects the computational cost of evaluating a tree.
    def node_count(self):
        # Terminal nodes contribute exactly one to the count, so we return immediately when the node has no children.
        # This base case keeps the recursion simple and efficient.
        if self.is_terminal():
            return 1
        # Internal nodes contribute one for themselves plus the counts of each child, so we sum the recursive results to obtain the full size.
        # The generator expression avoids creating temporary lists, making this method scalable even for large trees.
        return 1 + sum(child.node_count() for child in self.children)

    # iter_nodes yields every node in preorder along with metadata describing the parent, the child index, and the depth so that crossover and mutation can operate easily.
    # Providing this generator eliminates the need for repeated traversal code elsewhere in the module.
    def iter_nodes(self, parent=None, index=None, depth=0):
        # The current node is yielded first to preserve preorder traversal, which aligns with both serialization and standard GP operations.
        # Including the parent metadata allows callers to modify the tree structure without having to retrace paths from the root.
        yield (self, parent, index, depth)
        # We then recurse into each child while incrementing the depth so that every yielded tuple accurately reflects the node's position within the tree.
        # Enumerate supplies both the child index and the child reference, simplifying bookkeeping for callers that modify child pointers.
        for child_index, child in enumerate(self.children):
            yield from child.iter_nodes(self, child_index, depth + 1)

    # Implementing __deepcopy__ gives us precise control over how nodes are cloned, ensuring that each subtree copy maintains independence from the original.
    # This method supports Python's deepcopy protocol, letting other parts of the code duplicate trees safely without manual recursion.
    def __deepcopy__(self, memo):
        # Each child is deep copied recursively so that the resulting subtree shares no references with the original.
        # Passing the memo dictionary through prevents redundant work if the same node is encountered multiple times, although our trees are acyclic by construction.
        copied_children = [deepcopy(child, memo) for child in self.children]
        # We return a new TreeNode containing the same primitive and value but with cloned children, producing a faithful yet independent copy of the subtree.
        # This design ensures that mutation and crossover can manipulate the new subtree without risking unintended changes to the source individual.
        return TreeNode(self.primitive, self.value, copied_children)


# ParseTree wraps a root node together with the primitive sets so that serialization, mutation, and metrics all share a consistent interface.
# Encapsulating these responsibilities keeps the TreeGenotype class lightweight while still providing all the necessary tree-specific functionality.
class ParseTree:
    # The initializer stores the root node plus the terminal and nonterminal collections, which are needed when generating or deserializing subtrees later on.
    # Saving these sets with the tree avoids having to thread them through every function call, simplifying APIs throughout the module.
    def __init__(self, root, terminals, nonterminals):
        # The root node anchors the entire tree structure, so we store it directly without wrapping to keep traversal fast and straightforward.
        # Expecting a TreeNode instance here enforces consistency across all tree operations.
        self.root = root
        # Terminals are stored as an immutable tuple to prevent accidental modification and to preserve a deterministic order for random sampling.
        # Converting to a tuple also guarantees compatibility with random.choice, which expects a sequence.
        self.terminals = tuple(terminals)
        # Nonterminals receive the same treatment so that operator selection remains stable and easily reproducible.
        # Keeping the raw strings avoids translation layers and keeps serialization identical to the configuration files.
        self.nonterminals = tuple(nonterminals)

    # generate constructs a random tree following either the full or grow strategy, making it useful both during population initialization and mutation.
    # The classmethod form allows callers to request a ParseTree without first instantiating an empty tree, which keeps the API intuitive.
    @classmethod
    def generate(cls, method, max_depth, terminals, nonterminals):
        # We delegate the actual node construction to build_random_subtree so that tree generation logic stays consolidated in one place.
        # The helper returns a fully formed TreeNode hierarchy, which we then wrap in a ParseTree to attach the primitive metadata.
        root = build_random_subtree(method, max_depth, terminals, nonterminals)
        # Returning a new ParseTree gives callers a complete tree ready for evaluation or serialization with no additional setup.
        # This design also makes unit testing straightforward because the helper function can be exercised independently.
        return cls(root, terminals, nonterminals)

    # clone produces a deep copy of the tree so that genetic operators can modify the clone without affecting the original parent.
    # Returning another ParseTree instance ensures that the primitive metadata accompanies the cloned structure, preserving full functionality.
    def clone(self):
        # deepcopy on the root creates an independent copy of the entire tree structure in one call.
        # We reuse the stored primitive sets so that the clone behaves exactly like the original in terms of valid primitives.
        return ParseTree(deepcopy(self.root), self.terminals, self.nonterminals)

    # serialize converts the tree into the newline-delimited preorder format required by the assignment so that parse trees can be logged and reloaded.
    # Keeping the serialization logic inside the tree class prevents duplication and ensures that the representation stays synchronized with the structure.
    def serialize(self):
        # We accumulate serialized lines in a list and join them at the end because list appends are efficient and preserve node order accurately.
        # This approach also makes it easy to inspect intermediate results during debugging.
        lines = []

        # Nested helper for preorder traversal keeps the recursion local to this method, reducing the public surface area of the class.
        # Tracking the depth allows the helper to prepend the correct number of pipe characters to each line.
        def visit(node, depth):
            # Constant nodes print their numeric value to match the serialization requirements, so we convert the stored float to a string directly.
            # Prefixing the value with pipe characters records the node's depth exactly as described in the assignment.
            if node.primitive == 'C':
                lines.append('|' * depth + str(float(node.value)))
            else:
                # Non-constant nodes output their primitive name, again preceded by the appropriate number of pipe characters.
                # This representation is both human-readable and easy to parse during deserialization.
                lines.append('|' * depth + node.primitive)
            # After recording the current node we descend into each child, incrementing the depth so that nested primitives receive additional pipe characters.
            # Terminals have no children, so the loop exits immediately in those cases, satisfying the preorder traversal definition.
            for child in node.children:
                visit(child, depth + 1)

        # The traversal begins at the root with depth zero, ensuring that the root line contains no pipe characters.
        # Starting here guarantees that every node appears exactly once in the serialized output.
        visit(self.root, 0)
        # Joining the lines with newline characters yields the final serialization string without an unnecessary trailing newline.
        # Returning the string lets other modules write the representation to disk or transmit it over multiprocessing boundaries.
        return '\n'.join(lines)

    # node_count exposes the underlying node count via the root helper, making it easy for external code to apply parsimony pressure without digging into the tree structure.
    # This method is intentionally thin so that any future refinements to the counting logic automatically propagate everywhere.
    def node_count(self):
        # Returning the root's node_count keeps the public API simple and avoids duplicating traversal code.
        # This call provides the exact same metric used during parsimony calculations within the evolutionary algorithm.
        return self.root.node_count()

    # height mirrors node_count by delegating to the root, allowing other components to query the tree depth quickly.
    # This symmetry also keeps the codebase consistent, reducing cognitive load for future maintainers.
    def height(self):
        # Returning the root height avoids redundant traversal implementations and keeps the method implementation minimal.
        # The resulting integer includes the root level, matching the expectation set elsewhere in the module.
        return self.root.height()


# Helper utilities below provide common functionality for sensor evaluation and tree generation, keeping the main classes uncluttered.
# Consolidating these helpers also makes it easier to reuse the logic in future extensions such as ghost controllers or multi-agent experiments.

def find_pac_location(players):
    # We iterate through the dictionary entries to locate a key containing the letter 'm', which denotes Pac-Man regardless of numbering.
    # Returning as soon as we find a match keeps the search efficient even when many players are present, such as in multi-agent experiments.
    for name, location in players.items():
        # The naming convention used by the provided game logic reserves the letter 'm' for Pac-Man agents, so we exploit that convention here.
        # This design remains compatible with both the single-player key 'm' and the multi-player keys like 'm0'.
        if 'm' in name:
            return location
    # If no Pac-Man is present the state dictionary is malformed, so we raise an informative error to aid debugging immediately.
    # Raising a ValueError here prevents the evaluation routine from operating on meaningless data.
    raise ValueError('Pac-Man location not found in players dictionary')


def find_ghost_locations(players):
    # Ghost identifiers lack the letter 'm', so we collect all locations whose keys satisfy that property.
    # Returning a tuple keeps the result immutable while still supporting efficient iteration in the calling context.
    return tuple(location for name, location in players.items() if 'm' not in name)


def count_adjacent_walls(location, walls):
    # Unpacking the location coordinates into named variables clarifies the subsequent arithmetic and keeps the code readable.
    # Having explicit names for the axes also makes it easier to double-check the bounds logic.
    x, y = location
    # The width is determined by the number of columns in the map, which corresponds to the length of the outer list.
    # Storing this value avoids repeated len calls inside the loop, slightly improving performance.
    width = len(walls)
    # The height is the length of each column, so we inspect the first column to obtain it.
    # As with width, caching the value here keeps the loop body lean.
    height = len(walls[0])
    # We initialize the wall counter to zero so that we can increment it whenever a neighboring cell is blocked or out of bounds.
    # Using an integer counter keeps the function simple and efficient.
    count = 0
    # The four cardinal direction offsets are listed explicitly because the assignment restricts the wall sensor to immediate neighbors.
    # Representing them as tuples keeps the iteration straightforward and readable.
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    # We iterate over each direction, checking the corresponding cell and incrementing the counter when the neighbor is inaccessible.
    # This loop encapsulates the entire sensor computation, keeping the logic localized and easy to verify.
    for dx, dy in neighbors:
        # Adding the offset to the current coordinates yields the neighbor's position, which we need to evaluate next.
        # Using separate variables for the neighbor coordinates clarifies the subsequent bounds and wall checks.
        nx, ny = x + dx, y + dy
        # Positions outside the grid boundaries count as walls according to the assignment, so we detect them via range comparisons.
        # This check prevents us from accessing invalid indices in the walls structure.
        if nx < 0 or ny < 0 or nx >= width or ny >= height:
            # Incrementing the counter accounts for the implicit wall created by the boundary, matching the intended behavior of the sensor.
            # We continue to the next iteration immediately because no additional checks are needed for out-of-bounds cells.
            count += 1
            continue
        # For in-bounds neighbors we consult the walls structure directly to see whether the cell is blocked.
        # Since the walls structure stores booleans, we can use a simple truthiness test without extra conversions.
        if walls[nx][ny]:
            # Incrementing the counter adds the discovered wall to the total, ensuring the final result reflects all blocked directions.
            # No further action is required for this neighbor, so the loop proceeds to the next direction automatically.
            count += 1
    # Returning the final wall count provides the sensor value that higher-level functions can use in the parse tree evaluation.
    # The integer result integrates seamlessly with the arithmetic operators defined elsewhere in the tree.
    return count


def build_random_subtree(method, max_depth, terminals, nonterminals, current_depth=0):
    # Once the current depth reaches the limit we must produce a terminal node to satisfy the depth restriction imposed by the assignment.
    # Returning here guarantees that the recursion terminates and that generated trees never exceed the configured maximum depth.
    if current_depth == max_depth:
        return create_terminal_node(terminals)
    # The full method always selects a nonterminal until the final depth, producing perfectly balanced trees, so we choose from the nonterminal set explicitly.
    # Converting the method name to lowercase allows callers to pass either 'full' or 'Full' without issue, improving usability.
    if method.casefold() == 'full':
        primitive = random.choice(tuple(nonterminals))
    else:
        # The grow method may choose either a terminal or a nonterminal at intermediate depths, so we sample from the combined set here.
        # This approach yields structurally diverse trees that complement the strictly balanced trees produced by the full method.
        primitive = random.choice(tuple(terminals) + tuple(nonterminals))
    # If the selected primitive is a terminal we construct and return the corresponding leaf node immediately.
    # This branch ensures that grow-generated trees can terminate early on some branches, as intended by the method's definition.
    if primitive in terminals:
        return create_terminal_node([primitive])
    # When a nonterminal is selected we build both children recursively, incrementing the depth so that the depth constraint continues to apply properly.
    # Using a list comprehension highlights that both children are generated using the same method and depth settings, keeping the code concise.
    children = [build_random_subtree(method, max_depth, terminals, nonterminals, current_depth + 1) for _ in range(2)]
    # Finally we return a TreeNode representing the operator along with its freshly generated children.
    # This completes the recursive construction for this subtree, allowing higher levels to attach it appropriately.
    return TreeNode(primitive, children=children)


def create_terminal_node(terminals):
    # Terminals may be provided as any iterable, so we convert them to a tuple before sampling to ensure compatibility with random.choice.
    # Uniformly sampling from the provided set respects the user's configuration and keeps initialization unbiased.
    primitive = random.choice(tuple(terminals))
    # Constant terminals require special handling because each instance must receive an independently sampled numeric value.
    # For all other terminals we simply return a TreeNode with the primitive label and no children.
    if primitive == 'C':
        return create_constant_node()
    return TreeNode(primitive)


def create_constant_node():
    # random.uniform samples a floating-point number within the configured range, giving each constant node a unique value.
    # Sampling at node creation time aligns with the assignment's requirement that each constant is independent of all others.
    value = random.uniform(*CONSTANT_RANGE)
    # The returned TreeNode stores the 'C' primitive along with the sampled value, making it indistinguishable from other constants during evaluation and serialization.
    # Leaving the children list empty ensures that the rest of the code recognizes this node as a terminal.
    return TreeNode('C', value=value)


class TreeGenotype():
    # The initializer sets fitness and gene storage to None so that individuals begin in a clean, uninitialized state.
    # This mirrors the behavior from Assignment Series 1 and keeps the class compatible with the rest of the framework.
    def __init__(self):
        # Fitness starts as None because individuals have not yet been evaluated at construction time.
        # Using None rather than zero avoids implying that unevaluated individuals achieved a meaningful score.
        self.fitness = None
        # Genes also start as None, signaling that initialization must assign a ParseTree before evaluation occurs.
        # This default helps catch mistakes where an algorithm forgets to generate trees before calling fitness routines.
        self.genes = None

    # Population initialization constructs mu individuals whose trees follow the ramped half-and-half strategy described in the assignment.
    # Accepting **kwargs allows the method to receive terminal and nonterminal sets from the configuration file without hardcoding them here.
    @classmethod
    def initialization(cls, mu, depth_limit, **kwargs):
        # Terminals and nonterminals are extracted from the keyword arguments so that tree generation uses the caller's configuration consistently.
        # Converting to tuples ensures compatibility with random.choice and guards against accidental downstream modification.
        terminals = tuple(kwargs.get('terminals', ('G', 'P', 'F', 'W', 'C')))
        nonterminals = tuple(kwargs.get('nonterminals', ('+', '-', '*', '/', 'RAND')))
        # The initial population list is constructed by instantiating the class mu times so that each individual receives its own storage for genes and fitness values.
        # Using a list comprehension keeps the code concise and matches the style used in previous assignments.
        population = [cls() for _ in range(mu)]
        # Half of the population should use the full method while the remainder uses the grow method, so we compute the split point once up front.
        # Integer division handles odd mu values gracefully by assigning the extra individual to the grow half, which provides slightly more structural variety.
        split_index = mu // 2
        # We iterate over the individuals with their indices so that we can decide which generation method to apply to each one based on its position in the population list.
        # This deterministic assignment guarantees that exactly half of the individuals (rounded down) use the full method every time initialization runs.
        for index, individual in enumerate(population):
            # Each tree receives a depth limit sampled uniformly from the inclusive range [1, depth_limit] to satisfy the ramped depth distribution requirement.
            # Sampling per individual spreads the trees across different depths without needing complex bookkeeping.
            target_depth = random.randint(1, depth_limit)
            # The generation method alternates between full and grow according to the index relative to the split point so that the population adheres to the half-and-half structure.
            # Using a conditional expression keeps the logic compact while remaining easy to read.
            method = 'full' if index < split_index else 'grow'
            # ParseTree.generate constructs the random tree using the requested method and depth, returning a ParseTree that bundles the root with the primitive metadata.
            # Assigning the generated tree to the individual's genes readies the individual for evaluation later in the pipeline.
            individual.genes = ParseTree.generate(method, target_depth, terminals, nonterminals)
        # Returning the populated list gives the caller immediate access to the initialized individuals for evaluation or further processing.
        # The population now satisfies the ramped half-and-half specification, enabling the remainder of the algorithm to proceed correctly.
        return population

    # serialize produces the newline-delimited preorder representation used for logging and multiprocessing communication.
    # Delegating to the ParseTree keeps the method short and maintains a single authoritative serialization implementation.
    def serialize(self):
        # When genes are absent we return an empty string so that callers can detect uninitialized individuals gracefully.
        # This guard prevents attribute errors in situations where serialization is attempted before initialization.
        if self.genes is None:
            return ''
        # Otherwise we delegate to the ParseTree serialization, which already produces the required text format.
        # Returning the resulting string gives the caller the exact representation expected elsewhere in the assignment framework.
        return self.genes.serialize()

    # deserialize reconstructs the parse tree from the serialization format by walking the preorder listing and rebuilding the node relationships.
    # The implementation leverages the helper used earlier for node creation to keep constants and operators consistent with programmatically generated trees.
    def deserialize(self, serialization):
        # We split the incoming string on newline characters and discard empty lines so that the algorithm operates on a clean list of node descriptors.
        # Filtering out empty strings also makes the logic resilient to trailing newlines that might appear when reading from files.
        lines = [line for line in serialization.split('\n') if line]
        # If no lines remain the serialization was empty, so we clear the genes attribute and return early to leave the individual in a consistent state.
        # Returning immediately avoids indexing into an empty list, which would raise an error.
        if not lines:
            self.genes = None
            return
        # The first line describes the root node because the serialization uses preorder traversal, so we convert it into a TreeNode via the helper.
        # This node anchors the reconstructed tree and becomes the starting point for the subsequent stack-based attachment logic.
        root = deserialize_node(lines[0])
        # The stack keeps track of nodes awaiting children along with their depths so that we can attach new nodes to the correct parents as we process the serialization.
        # Initializing the stack with the root at depth zero reflects that we are currently positioned at the top of the tree.
        parent_stack = [(root, 0)]
        # We iterate over the remaining lines, each describing one node in preorder, and use the stored depth information to determine where each node belongs.
        # Processing lines sequentially ensures that parents are always encountered before their children, which makes the stack approach possible.
        for line in lines[1:]:
            # Blank lines are skipped so that accidental whitespace in serialized files does not introduce malformed nodes.
            # Continuing early keeps the rest of the loop focused on valid node descriptions.
            if not line:
                continue
            # Counting leading pipe characters gives us the depth of the current node, matching the serialization format exactly.
            # This depth value allows us to determine how far up the stack we need to climb to reach the appropriate parent.
            depth = line.count('|')
            # Stripping the pipe characters reveals the primitive string or numeric constant associated with the node.
            # Using lstrip preserves any non-pipe characters even if they include other punctuation.
            primitive_text = line.lstrip('|')
            # We pop the most recent entry from the stack because preorder traversal guarantees that this entry is either the parent or a sibling of the new node.
            # The accompanying depth tells us whether we need to climb further up the stack to find the actual parent.
            parent, parent_depth = parent_stack.pop()
            # This flag indicates whether the next available child slot should be the right child, which happens when we have already filled the left child of the current parent.
            # It defaults to False because we assume we are attaching a left child unless we discover otherwise while adjusting the stack.
            attach_right = False
            # We continue popping from the stack while the stored depth is greater than or equal to the depth of the new node, effectively moving up the tree until we reach the correct ancestor.
            # Each pop corresponds to finishing a subtree, so when we eventually find a shallower parent we know we should attach the new node as that parent's right child.
            while parent_stack and parent_depth >= depth:
                parent, parent_depth = parent_stack.pop()
                attach_right = True
            # The helper interprets the primitive text and returns a TreeNode configured either as a constant or an operator with placeholder children.
            # Using the helper ensures that deserialized nodes share the same structure and validation as nodes generated during initialization or mutation.
            node = deserialize_node(primitive_text)
            # If the new node should be attached as the right child we replace the parent's second child entry; otherwise we append it as the left child.
            # Operator nodes allocate two child slots during creation, so assigning to index one is safe for the right-child case.
            if attach_right:
                parent.children[1] = node
            else:
                # Nonterminal parents preallocate two child slots, so we assign the new node into the first slot when attaching a left child.
                # Terminal parents should never receive children, but the extra append fallback keeps the code safe during debugging if malformed trees appear.
                if parent.children:
                    parent.children[0] = node
                else:
                    parent.children.append(node)
            # We push the parent back onto the stack because it may still need additional children after this insertion.
            # We also push the new node along with its depth so that subsequent lines can attach descendants to it appropriately.
            parent_stack.extend([(parent, parent_depth), (node, depth)])
        # We preserve existing terminal and nonterminal sets when possible so that deserialized individuals match the configuration they were originally created with.
        # If no genes existed previously we fall back to the default primitive sets from the assignment instructions.
        terminals = getattr(self.genes, 'terminals', ('G', 'P', 'F', 'W', 'C'))
        nonterminals = getattr(self.genes, 'nonterminals', ('+', '-', '*', '/', 'RAND'))
        # Assigning a new ParseTree constructed from the reconstructed root and primitive sets completes the deserialization process.
        # The individual now contains a fully usable tree identical to the one described by the serialization string.
        self.genes = ParseTree(root, terminals, nonterminals)

    # recombine performs subtree crossover by cloning this individual's tree and then attempting to graft a randomly selected subtree from the mate while respecting the depth limit.
    # The implementation mirrors standard GP subtree crossover, ensuring compatibility with the expectations established in the assignment materials.
    def recombine(self, mate, depth_limit, **kwargs):
        # We instantiate the child using the same class so that any subclass-specific behavior carries over automatically.
        # Creating the child upfront allows us to populate its genes and return it regardless of whether crossover succeeds.
        child = self.__class__()
        # If either parent lacks genes we fall back to cloning the available parent's tree so that the child remains valid even in degenerate scenarios.
        # This guard prevents attribute errors and keeps debugging sessions manageable when partially initialized individuals are present.
        if self.genes is None or mate.genes is None:
            source = self.genes if self.genes is not None else mate.genes
            child.genes = source.clone() if source is not None else None
            return child
        # We start from a clone of this individual's tree so that recombination modifies the clone without affecting the original.
        # Cloning also ensures that the child's primitive sets match the parent, preserving compatibility with subsequent operations.
        child.genes = self.genes.clone()
        # All nodes in the child's tree are collected into a list so that we can sample a crossover point uniformly across the entire structure.
        # Including parent references and depths alongside the nodes simplifies the replacement process later.
        candidate_targets = list(child.genes.root.iter_nodes())
        # The mate's nodes are collected similarly so that every subtree from the mate has a chance to be selected as donor material.
        # Working with lists allows us to use random.choice for unbiased sampling.
        donor_nodes = list(mate.genes.root.iter_nodes())
        # We attempt crossover multiple times to increase the likelihood of finding a donor subtree that fits within the depth limit after insertion.
        # Limiting the number of attempts keeps the method efficient even when both trees are near the depth limit.
        for _ in range(10):
            # A random candidate from the child's tree provides the insertion point, and the tuple unpacking yields its parent metadata and depth.
            # Sampling uniformly over all nodes maintains the stochastic nature of GP crossover while giving every subtree an opportunity to participate.
            target_node, target_parent, target_index, target_depth = random.choice(candidate_targets)
            # We deep copy a random subtree from the mate to ensure that the donor subtree is independent of the mate after insertion.
            # Deep copying also prevents later mutations from unexpectedly altering the mate's tree.
            donor_subtree = deepcopy(random.choice(donor_nodes)[0])
            # The resulting depth after replacing the target is the target's depth plus the height of the donor subtree minus one, since height counts the root level as one.
            # We only accept the replacement if this depth respects the global depth limit provided by the caller.
            if target_depth + donor_subtree.height() - 1 <= depth_limit:
                # When the target node is the root there is no parent to update, so we simply replace the child's root with the donor subtree.
                # This branch ensures that crossover can still succeed even when the randomly chosen target happens to be the root.
                if target_parent is None:
                    child.genes.root = donor_subtree
                else:
                    # For non-root targets we update the parent's reference at the recorded child index so that the donor subtree occupies the same structural position.
                    # This maintains the left/right orientation of the tree, which can matter for non-commutative operators.
                    target_parent.children[target_index] = donor_subtree
                # After a successful replacement we exit the loop early because the child now contains a valid crossover result.
                # Breaking here prevents unnecessary additional attempts and keeps the method efficient.
                break
        # Returning the child provides the caller with the recombined individual ready for evaluation or further mutation.
        # Even if crossover failed to find a valid insertion within the attempt limit, the child still contains a clone of this individual's tree, preserving algorithm stability.
        return child

    # mutate applies subtree mutation by replacing a randomly selected node with a freshly generated subtree that fits within the remaining depth budget.
    # This mirrors the canonical subtree mutation operator described in the assignment and maintains genetic diversity.
    def mutate(self, depth_limit, **kwargs):
        # We construct the mutant using the same class to preserve compatibility and to allow subclasses to customize behavior if necessary.
        # The mutant starts as a clone of this individual so that the mutation modifies only the new individual.
        mutant = self.__class__()
        mutant.genes = self.genes.clone() if self.genes is not None else None
        # If cloning failed because the original had no genes we simply return the mutant, which mirrors the parent's state.
        # This guard keeps the method robust when invoked on individuals that have not been initialized yet.
        if mutant.genes is None:
            return mutant
        # We gather all nodes from the cloned tree along with their parent metadata so that we can choose a mutation point uniformly.
        # Storing the nodes in a list allows us to use random.choice without additional iterators.
        nodes = list(mutant.genes.root.iter_nodes())
        # Sampling a random entry determines which subtree will be replaced, giving every node an equal chance of being mutated.
        # The tuple unpacking provides the parent pointer, child index, and depth required for safe replacement.
        _, target_parent, target_index, target_depth = random.choice(nodes)
        # The remaining depth available for the new subtree equals the global depth limit minus the depth of the mutation point, with a minimum of one to keep the generator valid.
        # This calculation ensures that the mutated tree never exceeds the configured depth limit.
        remaining_depth = max(1, depth_limit - target_depth)
        # We generate a new subtree using the grow method to introduce structural variety and to align with common GP mutation strategies.
        # Passing the remaining depth keeps the new subtree within the allowed height when inserted at the selected node.
        new_subtree = build_random_subtree('grow', remaining_depth, mutant.genes.terminals, mutant.genes.nonterminals)
        # If the selected node is the root we replace the entire tree directly because there is no parent pointer to update.
        # This case allows mutations to dramatically alter the tree structure when necessary.
        if target_parent is None:
            mutant.genes.root = new_subtree
        else:
            # For non-root nodes we replace the corresponding child reference on the parent with the new subtree, leaving the rest of the tree untouched.
            # Updating via the stored index maintains the original left/right orientation even after mutation.
            target_parent.children[target_index] = new_subtree
        # Returning the mutant completes the mutation process, yielding an individual that differs from the parent by at least one randomly generated subtree.
        # This approach encourages exploration of the search space while respecting the global depth constraint.
        return mutant


# Deserializing nodes requires converting primitive strings back into TreeNode instances, so this helper centralizes that logic for reuse across the module.
# Keeping the logic in one place prevents discrepancies between root creation and child creation, which simplifies maintenance and reduces the risk of subtle bugs.
def deserialize_node(text):
    # We first attempt to interpret the text as a floating-point number so that constant terminals printed as numeric strings can be reconstructed automatically.
    # Wrapping the conversion in a try/except block allows us to fall back gracefully when the text represents an operator or sensor primitive instead of a constant.
    try:
        value = float(text)
    except ValueError:
        # When conversion fails we treat the text as a primitive name and create a node with no children, which is appropriate for both sensors and operators prior to attachment.
        # The caller will assign children later for operator nodes, so we initially leave the child list empty here.
        node = TreeNode(text)
    else:
        # Successful conversion indicates a constant terminal, so we build a node with the 'C' primitive and store the parsed numeric value for consistent evaluation behavior.
        # This mirrors the structure created by create_constant_node, ensuring that deserialized constants behave identically to freshly generated ones.
        node = TreeNode('C', value=value)
    # Operator primitives require two child slots, so we preallocate a list containing two None entries to simplify later assignment in the deserialization routine.
    # Terminals retain an empty child list, which correctly identifies them as leaves for evaluation, serialization, and genetic operators.
    if node.primitive in ('+', '-', '*', '/', 'RAND'):
        node.children = [None, None]
    return node

