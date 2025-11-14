
# fitness.py

import gpac
import random
from functools import cache
from math import inf

# This helper identifies the TreeNode root associated with whichever controller object is supplied so that the action-selection logic can always evaluate a consistent interface. Handling the conversion in one place prevents duplication and makes it easy to extend support for new controller wrappers later on.
# I intentionally check for the most specific attribute first because TreeGenotype instances expose their trees through the genes attribute, and resolving that path early avoids unnecessary fallback logic.
def _extract_tree_root(controller):
    # We first look for a populated genes attribute, which indicates the caller passed in a TreeGenotype individual ready for evaluation. Guarding against a None value here prevents attribute errors when unevaluated individuals slip through during debugging.
    # Returning the associated parse tree's root gives us direct access to the TreeNode API that already implements recursive evaluation of GP trees.
    if hasattr(controller, 'genes') and controller.genes is not None:
        # Yielding the root node from the parse tree keeps the downstream code simple because all decision-making logic works with TreeNode objects directly. This also ensures we reuse the exact tree structure produced during initialization or mutation without extra cloning.
        return controller.genes.root
    # When the controller itself is a ParseTree, the root attribute already references the TreeNode we need, so we can return it immediately. Supporting this path lets notebook experiments pass parse trees without wrapping them inside a TreeGenotype instance.
    # The additional not-None guard here prevents failures when a partially constructed ParseTree slips through due to serialization bugs.
    if hasattr(controller, 'root') and controller.root is not None:
        # Returning the root TreeNode directly allows the evaluation code to remain agnostic about whether it received a ParseTree wrapper. This keeps compatibility with helper utilities that prefer to work with ParseTree objects explicitly.
        return controller.root
    # Some tests pass the TreeNode itself, so we accept any object exposing an evaluate method as already being the desired root. Relying on duck typing in this branch maximizes flexibility while still ensuring the returned object can perform the necessary computation.
    # This fallback is particularly useful when unit tests construct minimal trees manually without creating full genotype objects.
    if hasattr(controller, 'evaluate'):
        # Returning the controller unchanged lets callers provide bare TreeNode instances without additional ceremony. The evaluate method signature matches what our action-selection logic expects, so no further adaptation is required.
        return controller
    # If none of the above conditions succeeded, the controller is incompatible with our evaluation pipeline, so we raise an informative error. Failing fast like this prevents the game loop from silently reverting to random play, which would obscure bugs during experimentation.
    # The error message also guides future developers toward the supported controller interfaces, reducing troubleshooting time.
    raise TypeError('Unsupported controller type for GPac action selection')


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Fitness function that plays a game using the provided pac_controller
# with optional ghost controller and game map specifications.
# Returns Pac-Man score from a full game as well as the game log.
def play_GPac(pac_controller, ghost_controller=None, game_map=None, score_vector=False, **kwargs):
    game_map = parse_map(game_map)
    game = gpac.GPacGame(game_map, **kwargs)

    # Game loop, representing one turn.
    while not game.gameover:
        # Evaluate moves for each player.
        for player in game.players:
            actions = game.get_actions(player)
            s_primes = game.get_observations(actions, player)
            selected_action_idx = None

            # Select Pac-Man action(s) using provided strategy.
            if 'm' in player:
                if pac_controller is None:
                    # Random Pac-Man controller.
                    selected_action_idx = random.randrange(len(actions))

                else:
                    '''
                    ####################################
                    ###   YOUR 2a CODE STARTS HERE   ###
                    ####################################
                    '''
                    # We resolve the controller's parse tree root up front so that each candidate action uses the same TreeNode interface for scoring. Performing this lookup once per decision avoids redundant attribute checks inside the evaluation loop.
                    controller_root = _extract_tree_root(pac_controller)
                    # The best score tracker starts at negative infinity so that any legitimate evaluation value will supersede it immediately. Choosing -inf also ensures we never accidentally favor an uninitialized default over a real action score.
                    best_score = -inf
                    # We store the index of the current best action in this variable so the outer logic can register the move after all evaluations complete. Initializing to zero provides a valid fallback when multiple actions tie at the sentinel value.
                    best_index = 0
                    # Each hypothetical successor state produced by the environment gets evaluated so that the controller can compare all legal moves fairly. Enumerating the list delivers both the state and the action index, which keeps the bookkeeping clean and efficient.
                    for idx, candidate_state in enumerate(s_primes):
                        # A fresh cache dictionary is created for every state so that memoized sensor readings never leak between different hypothetical futures. This isolation is critical because the same sensor primitive can yield different values for different candidate states.
                        state_cache = dict()
                        # We evaluate the program tree on the candidate state to obtain a numerical desirability score for the associated action. Supplying the cache enables memoization within the tree evaluation, which keeps the computation fast even for large trees.
                        action_score = controller_root.evaluate(candidate_state, state_cache)
                        # Whenever the new score beats our current best, we adopt the new value and remember the corresponding index. Using a strict greater-than comparison keeps the first discovered maximum when multiple actions tie, which matches the deterministic behavior of Python's max.
                        if action_score > best_score:
                            # Updating the best score here ensures that subsequent comparisons use the most recent elite value. This design maintains numerical stability while keeping the loop logic easy to follow during debugging.
                            best_score = action_score
                            # Recording the index at the same time guarantees the selected action matches the stored best score. Keeping the two assignments adjacent minimizes the risk of desynchronization if future edits change the comparison policy.
                            best_index = idx
                    # After evaluating every legal action we commit to the best index so the game can apply the chosen move. This single assignment serves as the bridge between the GP decision process and the GPac engine's action registration.
                    selected_action_idx = best_index

                    # You may want to uncomment these print statements for debugging.
                    # print(selected_action_idx)
                    # print(actions)

                    '''
                    ####################################
                    ###    YOUR 2a CODE ENDS HERE    ###
                    ####################################
                    '''

            # Select Ghost action(s) using provided strategy.
            else:
                if ghost_controller is None:
                    # Random Ghost controller.
                    selected_action_idx = random.randrange(len(actions))

                else:
                    '''
                    ####################################
                    ###   YOUR 2c CODE STARTS HERE   ###
                    ####################################
                    '''
                    # 2c TODO: Score all of the states stored in s_primes by evaluating your tree

                    # 2c TODO: Assign index of state with the best score to selected_action_idx.
                    selected_action_idx = None

                    # You may want to uncomment these print statements for debugging.
                    # print(selected_action_idx)
                    # print(actions)
                    
                    '''
                    ####################################
                    ###    YOUR 2c CODE ENDS HERE    ###
                    ####################################
                    '''

            game.register_action(actions[selected_action_idx], player)
        game.step()
    if not score_vector:
        return game.score, game.log
    return game.score, game.log, game.score_vector


# Function for parsing map contents.
# Note it is cached, so modifying a file requires a kernel restart.
@cache
def parse_map(path_or_contents):
    if not path_or_contents:
        # Default generic game map, with a cross-shaped path.
        size = 21
        game_map = [[True for __ in range(size)] for _ in range(size)]
        for i in range(size):
            game_map[0][i] = False
            game_map[i][0] = False
            game_map[size//2][i] = False
            game_map[i][size//2] = False
            game_map[-1][i] = False
            game_map[i][-1] = False
        return tuple(tuple(y for y in x) for x in game_map)

    if isinstance(path_or_contents, str):
        if '\n' not in path_or_contents:
            # Parse game map from file path.
            with open(path_or_contents, 'r') as f:
                lines = f.readlines()
        else:
            # Parse game map from a single string.
            lines = path_or_contents.split('\n')
    elif isinstance(path_or_contents, list) and isinstance(path_or_contents[0], str):
        # Parse game map from a list of strings.
        lines = path_or_contents[:]
    else:
        # Assume the game map has already been parsed.
        return path_or_contents

    for line in lines:
        line.strip('\n')
    firstline = lines[0].split(' ')
    width, height = int(firstline[0]), int(firstline[1])
    game_map = [[False for y in range(height)] for x in range(width)]
    y = -1
    for line in lines[1:]:
        for x, char in enumerate(line):
            if char == '#':
                game_map[x][y] = True
        y -= 1
    return tuple(tuple(y for y in x) for x in game_map)
