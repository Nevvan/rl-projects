"""
Gridworld skeleton — flat integer state IDs + NumPy arrays.
States are IDs 0 .. size*size - 1 (row-major: id = row*size + col).
Each function/method lists only its inputs and required outputs.
"""

import numpy as np


class GridWorld:
    def __init__(self, size=4):
        """
        Input: size (int, grid is size x size)
        Sets up: self.size, self.n_states, self.states (all ids),
        self.terminal_states (set of ids), self.non_terminal_states (list),
        self.actions (dict: action name -> (row_delta, col_delta))
        """
        self.size = size
        self.n_states = size * size
        self.states = [s for s in range(0, self.n_states)]
        self.terminal_states = {0, size * size - 1}
        self.non_terminal_states = [s for s in self.states if s not in self.terminal_states]
        self.actions = {"up": (-1, 0), "down": (1, 0), "right": (0, 1), "left": (0, -1)}

    def id_to_rc(self, state_id):
        """
        Input: state_id (int)
        Output: (row, col) tuple
        """
        row = state_id // self.size
        col = state_id % self.size
        return (row, col)

    def rc_to_id(self, row, col):
        """
        Input: row (int), col (int)
        Output: state_id (int)
        """
        state_id = row * self.size + col
        return state_id

    def is_terminal(self, state_id):
        """
        Input: state_id (int)
        Output: bool
        """
        if state_id in self.terminal_states:
            return True
        else:
            return False

    def step(self, state_id, action):
        """
        Input: state_id (int), action (str, key into self.actions)
        Output: (next_state_id (int), reward (float))
        """
        row, col = self.id_to_rc(state_id)
        row_delta, col_delta = self.actions[action]

        if (row == 0 and action == "up") or (row == self.size - 1 and action == "down") or \
           (col == 0 and action == "left") or (col == self.size - 1 and action == "right"):
            new_row, new_col = row, col
        else:
            new_row, new_col = row + row_delta, col + col_delta

        new_state_id = self.rc_to_id(new_row, new_col)
        reward = -1

        return new_state_id, reward

    def get_all_states(self):
        """
        Input: none
        Output: list[int] of all state ids
        """
        return self.states

    def get_actions(self):
        """
        Input: none
        Output: list[str] of action names
        """
        return list(self.actions.keys())


class Policy:
    def __init__(self, grid_world):
        """
        Input: grid_world (GridWorld instance)
        Sets up: self.policy, a NumPy array of shape (n_states, n_actions),
        representing action probabilities per state
        """
        n_actions = len(grid_world.actions.keys())
        self.policy = np.full((grid_world.n_states, n_actions), 1 / n_actions)

    def get_action_probs(self, state_id):
        """
        Input: state_id (int)
        Output: NumPy array of shape (n_actions,), probabilities summing to 1
        """
        action_probs = self.policy[state_id]
        return action_probs

    def set_greedy_action(self, state_id, action_index):
        """
        Input: state_id (int), action_index (int)
        Output: none (mutates self.policy in place)
        """
        greedy_action = np.zeros_like(self.policy[state_id])
        greedy_action[action_index] = 1
        self.policy[state_id] = greedy_action


class PolicyIterationAgent:
    def __init__(self, grid_world, policy, gamma=1.0, theta=1e-4):
        """
        Input: grid_world (GridWorld), policy (Policy), gamma (float), theta (float)
        Sets up: self.V, NumPy array of shape (n_states,), initialized to zeros
        """
        self.V = np.zeros((grid_world.n_states))
        self.gridworld = grid_world
        self.policy = policy
        self.gamma = gamma
        self.theta = theta
        self.actions_list = self.gridworld.actions.keys()

    def policy_evaluation(self):
        """
        Input: none (uses self.grid_world, self.policy, self.V, self.gamma, self.theta)
        Output: none (updates self.V in place until convergence)
        """
        delta = 10
        while delta > self.theta:
            delta = 0
            for state in self.gridworld.non_terminal_states:
                v = self.V[state]
                # --- old version (bug: self-referencing actions read a
                # partially-updated self.V[state] mid-accumulation) ---
                # self.V[state] = 0
                # for action_index, action in enumerate(self.actions_list):
                #     next_state, reward = self.gridworld.step(state, action)
                #     self.V[state] += self.policy.get_action_probs(state)[action_index] * \
                #         (reward + self.gamma * self.V[next_state])
                # delta = max((np.abs(self.V[state] - v), delta))

                # --- fixed version: accumulate into a local variable,
                # assign to self.V[state] only once the full sum is done ---
                new_v = 0
                for action_index, action in enumerate(self.actions_list):
                    next_state, reward = self.gridworld.step(state, action)
                    new_v += self.policy.get_action_probs(state)[action_index] * \
                        (reward + self.gamma * self.V[next_state])
                self.V[state] = new_v
                delta = max((np.abs(new_v - v), delta))

    def policy_improvement(self):
        """
        Input: none (uses self.grid_world, self.policy, self.V, self.gamma)
        Output: bool — True if policy unchanged (stable), False otherwise
        (also mutates self.policy in place)
        """
        policy_stable = True
        for state in self.gridworld.non_terminal_states:
            action_probs = self.policy.get_action_probs(state).copy()

            q = np.zeros_like(action_probs)
            for action_index, action in enumerate(self.actions_list):
                next_state, reward = self.gridworld.step(state, action)
                q[action_index] = reward + self.gamma * self.V[next_state]

            self.policy.set_greedy_action(state, np.argmax(q))
            if not np.array_equal(action_probs, self.policy.policy[state]):
                policy_stable = False

        return policy_stable

    def run(self, max_iterations=100):
        """
        Input: max_iterations (int)
        Output: (V (NumPy array), policy (Policy instance))
        """
        iter = 0
        policy_stable = False
        while iter < max_iterations and not policy_stable:
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            iter += 1

        return self.V, self.policy


def print_values(V, size=4):
    """
    Input: V (NumPy array, shape (size*size,)), size (int)
    Output: none (prints grid to console)
    """
    grid = V.reshape(size, size)
    for row in grid:
        print(" ".join(f"{val:6.2f}" for val in row))


def print_policy(policy_obj, grid_world, size=4):
    """
    Input: policy_obj (Policy), grid_world (GridWorld), size (int)
    Output: none (prints grid of action symbols to console)
    """
    symbols = {"up": "^", "down": "v", "left": "<", "right": ">"}
    action_names = list(grid_world.actions.keys())

    grid_symbols = []
    for state_id in range(grid_world.n_states):
        if grid_world.is_terminal(state_id):
            grid_symbols.append("T")
        else:
            best_action_index = np.argmax(policy_obj.get_action_probs(state_id))
            best_action = action_names[best_action_index]
            grid_symbols.append(symbols[best_action])

    grid_symbols = np.array(grid_symbols).reshape(size, size)
    for row in grid_symbols:
        print(" ".join(f"{s:>2}" for s in row))


if __name__ == "__main__":
    grid_world = GridWorld(size=4)
    policy = Policy(grid_world)
    agent = PolicyIterationAgent(grid_world, policy, gamma=1.0, theta=1e-4)

    V, final_policy = agent.run()

    print("Values:")
    print_values(V, size=4)

    print("\nPolicy:")
    print_policy(final_policy, grid_world, size=4)