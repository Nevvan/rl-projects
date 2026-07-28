"""
Mountain Car skeleton — continuous state (position, velocity), linear function
approximation, semi-gradient Sarsa.
Mirrors the structure of the Gridworld skeleton: env class, policy/action-selection
class, function-approximator class, agent class with the same
env / policy / step->update loop shape.
"""

import numpy as np


class MountainCar:
    def __init__(self, min_position=-1.2, max_position=0.6, max_speed=0.07, goal_position=0.5):
        """
        Input: min_position, max_position, max_speed, goal_position (floats, env bounds)
        Sets up: self.min_position, self.max_position, self.max_speed, self.goal_position,
        self.actions (dict: action name -> force direction, e.g. {"reverse": -1, "none": 0, "forward": 1}),
        self.gravity (float, e.g. -0.0025)
        """

        self.min_position = min_position
        self.max_position = max_position
        self.max_speed = max_speed
        self.goal_position = goal_position
        self.actions = {"reverse": -1, "none": 0, "forward": 1}
        self.gravity = -0.0025
        self.force = 0.0005

    def reset(self, range_start=-0.3, range_end=0.2):
        """
        Input: range_start, range_end (floats, position sampled uniformly from this range),
        velocity always starts at 0
        Output: state (tuple: (position, velocity))
        """
        position = np.random.random()*(range_end-range_start)+range_start
        velocity = np.random.random()*self.max_speed
        return position,velocity

    def step(self, state, action):
        """
        Input: state (tuple: (position, velocity)), action (str, key into self.actions)
        Output: (next_state (tuple), reward (float), done (bool))
        Physics: velocity += force*self.actions[action] + cos(3*position)*self.gravity,
        velocity clipped to [-max_speed, max_speed], position += velocity,
        position clipped to [min_position, max_position] (velocity zeroed if
        position hits min_position), reward is -1 per step until terminal.
        """
        position = state[0]
        velocity = state[1]
        done = False
        reward= -1
        action_taken = self.actions[action]

        velocity += self.force*action_taken + np.cos(3*position)*self.gravity

        if velocity > self.max_speed:
            velocity = self.max_speed
        elif velocity < -self.max_speed:
            velocity = -self.max_speed

        position += velocity

        if position < self.min_position:
            velocity = 0
            position = self.min_position
        elif position > self.max_position:
            position = self.max_position

        if position > self.goal_position:
            done = True

        return (position,velocity),reward,done
       

    def is_terminal(self, state):
        """
        Input: state (tuple: (position, velocity))
        Output: bool — True if position >= self.goal_position
        """
        if state[0] >= self.goal_position:
            return True
        else:
            return False

    def get_actions(self):
        """
        Input: none
        Output: list[str] of action names
        """
        action_list = list(self.actions.keys())
        return action_list


class TileCoder:
    def __init__(self, state_bounds, action_list, num_tilings=8, tiles_per_dim=8):
        """
        Input: state_bounds (dict: dim_name -> (low, high), e.g. {"position": (...), "velocity": (...)}),
        action_list (list[str], the environment's action names — needed so TileCoder
        can index actions into the feature vector), num_tilings (int), tiles_per_dim (int)
        Sets up: self.action_list, self.num_tilings, self.tiles_per_dim, self.state_bounds,
        self.position_width, self.velocity_width, self.tilings (list of per-tiling offsets),
        self.n_features (int, total feature vector length = num_tilings * tiles_per_dim^n_dims * n_actions)
        """
        self.action_list = action_list
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim
        self.state_bounds = state_bounds
        self.position_width = (state_bounds["position"][1] - state_bounds["position"][0])/tiles_per_dim
        self.velocity_width = (state_bounds["velocity"][1] - state_bounds["velocity"][0])/tiles_per_dim
        position_step_size = self.position_width/num_tilings
        velocity_step_size = self.velocity_width/num_tilings
        position_stride = 3
        velocity_stride = 5
        self.tilings = []
        for i in range(0,num_tilings):
            position_offset = ((i*position_stride)%num_tilings)*position_step_size
            velocity_offset = ((i*velocity_stride)%num_tilings)*velocity_step_size
            self.tilings.append((position_offset,velocity_offset))
        self.n_features = num_tilings * tiles_per_dim** 2 * len(action_list)
        

    def get_active_tiles(self, state, action):
        """
        Input: state (tuple: (position, velocity)), action (str)
        Output: list[int] — indices of active (nonzero) features in the full
        feature vector for this (state, action) pair (one active index per tiling)
        """
        state_ids = []
        action_id = self.action_list.index(action)

        for i in range(len(self.tilings)):
            corrected_state = [state[0]-self.tilings[i][0],state[1]-self.tilings[i][1]]
            
            if corrected_state[1] > self.state_bounds["velocity"][1]:
                corrected_state[1] = self.state_bounds["velocity"][1]
            elif corrected_state[1] < self.state_bounds["velocity"][0]:
                        corrected_state[1] = self.state_bounds["velocity"][0]

            if corrected_state[0] > self.state_bounds["position"][1]:
                corrected_state[0] = self.state_bounds["position"][1]
            elif corrected_state[0] < self.state_bounds["position"][0]:
                corrected_state[0] = self.state_bounds["position"][0]

            position_index = corrected_state[0]//self.position_width
            velocity_index = corrected_state[1]//self.velocity_width
            state_id = int(((((position_index*self.tiles_per_dim+velocity_index)*self.num_tilings+i)*len(self.action_list))+action_id))
            state_ids.append(state_id)

        return state_ids
    
    def featurize(self, state, action):
        """
        Input: state (tuple), action (str)
        Output: NumPy array of shape (n_features,), sparse binary feature vector
        (1s at active tile indices from get_active_tiles, 0 elsewhere)
        """
        state_ids = self.get_active_tiles(state,action)
        feature_vector = np.zeros(self.n_features)

        for i in state_ids:
            feature_vector[i] = 1

        return feature_vector


class LinearQFunction:
    def __init__(self, tile_coder, actions):
        """
        Input: tile_coder (TileCoder instance), actions (list[str])
        Sets up: self.tile_coder, self.actions, self.w (NumPy array of shape
        (n_features,), initialized to zeros)
        """
        self.tile_coder = tile_coder
        self.w = np.zeros(tile_coder.n_features)
        self.actions = actions

    def q_value(self, state, action):
        """
        Input: state (tuple), action (str)
        Output: float — Q(s,a;w), dot product of feature vector and self.w
        (equivalently: sum of self.w at the active tile indices, since features are binary)
        """
        feature_vector = self.tile_coder.featurize(state,action)        
        q = np.dot(feature_vector,self.w)

        return q

    def q_values_all_actions(self, state):
        """
        Input: state (tuple)
        Output: NumPy array of shape (n_actions,) — Q(s,a;w) for every action at this state
        """
        q_state_vector = np.zeros(len(self.actions))

        for i in range(len(self.actions)):
            q_state_vector[i] = self.q_value(state,self.actions[i])

        return q_state_vector

    def update(self, state, action, target, alpha):
        """
        Input: state (tuple), action (str), target (float, TD target), alpha (float, step size)
        Output: none (updates self.w in place via semi-gradient step:
        w <- w + alpha * (target - Q(s,a;w)) * grad_w Q(s,a;w), where grad_w Q
        is just the feature vector itself since Q is linear in w — in practice
        this only touches the active tile indices, not the full vector)
        """
        """self.w = self.w + alpha * (target - self.q_value(state,action)) * self.tile_coder.featurize(state,action)"""

        active_tiles = self.tile_coder.get_active_tiles(state,action)
        q = self.q_value(state,action)
        for i in active_tiles:
            self.w[i] += alpha * (target - q)

class EpsilonGreedyPolicy:
    def __init__(self, q_function, epsilon=0.1):
        """
        Input: q_function (LinearQFunction instance), epsilon (float, exploration rate)
        Sets up: self.q_function, self.epsilon
        """
        self.q_function = q_function
        self.epsilon = epsilon

    def select_action(self, state):
        """
        Input: state (tuple)
        Output: action (str) — uniform-random action w.p. self.epsilon,
        else argmax_a Q(s,a;w) via self.q_function.q_values_all_actions(state)
        """
        if np.random.random() > self.epsilon:
            q_state_vector = self.q_function.q_values_all_actions(state)
            action_index = np.argmax(q_state_vector)

            return self.q_function.actions[action_index]

        else:
            action_index = np.random.randint(0,len(self.q_function.actions))  

            return self.q_function.actions[action_index]
        
    def decay_epsilon(self, decay_rate):
        """
        Input: decay_rate (float)
        Output: none (updates self.epsilon in place, e.g. self.epsilon *= decay_rate)
        """
        self.epsilon *= decay_rate


class SemiGradientSarsaAgent:
    def __init__(self, env, q_function, policy, gamma=1.0, alpha=0.1):
        """
        Input: env (MountainCar instance), q_function (LinearQFunction),
        policy (EpsilonGreedyPolicy), gamma (float, discount), alpha (float, step size)
        Sets up: self.env, self.q_function, self.policy, self.gamma, self.alpha,
        self.episode_lengths (list, tracks steps-to-goal per episode for diagnostics)
        """
        self.env = env
        self.q_function = q_function
        self.policy = policy
        self.alpha = alpha
        self.gamma = gamma
        self.episode_lengths = []

    def run_episode(self, max_steps=1000):
        """
        Input: max_steps (int, cutoff to bound episode length before learning kicks in)
        Output: episode_length (int, number of steps taken this episode)
        (also updates self.q_function.w in place) — the on-policy Sarsa loop:
        """
        s = self.env.reset()
        a = self.policy.select_action(s)
        done = False
        step_count = 0

        while step_count < max_steps:
            next_s,r,done = self.env.step(s,a)

            if done:
                target = r
                self.q_function.update(s,a,target,self.alpha)
                return step_count+1
            else:
                next_a = self.policy.select_action(next_s)
                target = r + self.gamma * self.q_function.q_value(next_s,next_a)
                self.q_function.update(s,a,target,self.alpha)

            s = next_s
            a = next_a
            step_count +=1
        else:
            return max_steps
            

    def train(self, epsilon_decay_rate, num_episodes=500, max_steps=1000):
        """
        Input: epsilon_decay_rate (float, multiplicative decay applied to policy.epsilon
        after every episode), num_episodes (int), max_steps (int, per-episode cutoff)
        Output: self.episode_lengths (list[int]) — full training history
        (also mutates self.q_function.w and self.policy.epsilon in place across episodes)
        """
        episode_count = 0

        while episode_count < num_episodes:
            self.episode_lengths.append(self.run_episode(max_steps))
            self.policy.decay_epsilon(epsilon_decay_rate)
            episode_count+=1
        else:
            return self.episode_lengths

def plot_episode_lengths(episode_lengths):
    """
    Input: episode_lengths (list[int])
    Output: none (plots learning curve: episode length / steps-to-goal vs episode number)
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(episode_lengths) + 1), episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Steps to goal")
    plt.title("Mountain Car — Episode Length over Training")
    plt.grid(True)
    plt.show()


def plot_value_function(q_function, env, resolution=50):
    """
    Input: q_function (LinearQFunction), env (MountainCar), resolution (int, grid density)
    Output: none (plots -max_a Q(s,a;w) as a 2D surface/heatmap over position x velocity —
    the standard Mountain Car "cost-to-go" diagnostic visualization)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3D projection)

    positions = np.linspace(env.min_position, env.max_position, resolution)
    velocities = np.linspace(-env.max_speed, env.max_speed, resolution)
    P, Vel = np.meshgrid(positions, velocities)

    cost_to_go = np.zeros_like(P)
    for i in range(resolution):
        for j in range(resolution):
            state = (P[i, j], Vel[i, j])
            q_values = q_function.q_values_all_actions(state)
            cost_to_go[i, j] = -np.max(q_values)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(P, Vel, cost_to_go, cmap="viridis")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("Cost-to-go (-max_a Q)")
    ax.set_title("Mountain Car — Learned Cost-to-Go Function")
    plt.show()


if __name__ == "__main__":
    env = MountainCar()
    action_list = env.get_actions()

    tile_coder = TileCoder(
        state_bounds={
            "position": (env.min_position, env.max_position),
            "velocity": (-env.max_speed, env.max_speed),
        },
        action_list=action_list,
    )
    q_function = LinearQFunction(tile_coder, action_list)
    policy = EpsilonGreedyPolicy(q_function, epsilon=0.1)
    agent = SemiGradientSarsaAgent(env, q_function, policy, gamma=1.0, alpha=0.1)

    episode_lengths = agent.train(epsilon_decay_rate=0.99, num_episodes=2500)

    plot_episode_lengths(episode_lengths)
    plot_value_function(q_function, env)

    """print(episode_lengths)"""