from typing import Union, List, Tuple
from .env_base import Env  # Assuming the Env class is in a file named env.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from visualization.heat_map import plot_heatmap_with_arrows as heat_map
from stochastic_reward import StochasticReward

class GridWorld(Env):  # Inheriting from Env
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(
        self,
        height: int,
        width: Union[int, None] = None,
        actions: Union[List[int], None] = None,
        rewards: Union[List[float], None] = None,
        proba: float = 0.8,
        terminal_states: Union[List[tuple], None] = None,
        barrier_states: Union[List[tuple], None] = None,
        initial_state: Union[tuple, None] = None,
        task = None,
        distinct_rewards: int = 3
    ):
        
        """
        Initialize a GridWorld object.

        Parameters:
        - height (int): The height of the grid.
        - width (Optional[int]): The width of the grid. If not provided, defaults to height.
        - actions (Optional[List[int]]): List of possible actions (default: [0, 1, 2, 3]).
        - rewards (Optional[List[float]]): The rewards grid. If not provided, defaults to zeros.
        - proba (float): Probability of actions succeeding (default: 0.8).
        - terminal_states (Optional[List[tuple]]): List of terminal states (default: None).
        - barrier_states (Optional[List[tuple]]): List of barrier states (default: None).
        - initial_state (Optional[tuple]): Initial state (default: None).
        - task (Optional[str]): Task to be performed by the agent (default: None).
        - distinct_rewards (int): Number of distinct rewards (default: 3).
        """
            
        # Assert height and width
        assert isinstance(height, int) and height > 0, "Height should be a positive integer"
        self.height = height
        if width is not None:
            assert isinstance(width, int) and width > 0, "Width should be a positive integer"
            self.width = width
        else:
            self.width = self.height  # Set width as height if not specified

        # Actions 
        self.actions = actions if actions is not None else [self.UP, self.RIGHT, self.DOWN, self.LEFT]

        # Number of states and actions
        self.n_states = self.height * self.width
        self.n_actions = len(self.actions)

        # Transition probabilities
        assert 0 <= proba <= 1, "Probability should be between 0 and 1"
        self.transition_probabilities = [proba] + [(1 - proba) / (len(self.actions) - 2)] * (len(self.actions) - 2) + [0]

        # Terminal and barrier states
        if terminal_states is not None:
            assert all(isinstance(state, tuple) and len(state) == 2 and 0 <= state[0] < self.width and 0 <= state[1] < self.height for state in terminal_states), "Terminal states should be tuples of length 2 within grid bounds"
        else:
            terminal_states = []
        self.terminal_states = terminal_states

        if barrier_states is not None:
            assert all(isinstance(state, tuple) and len(state) == 2 and 0 <= state[0] < self.width and 0 <= state[1] < self.height for state in barrier_states), "Barrier states should be tuples of length 2 within grid bounds"
        else:
            barrier_states = []
        self.barrier_states = barrier_states

        self.initial_state = initial_state

        # Initialize rewards grid
        if rewards is None:
            self.rewards = [[StochasticReward(-1, deterministic = True, mean = - 1) for _ in range(self.width)] for _ in range(self.height)]

            # Update terminal states to have 0 reward
            if self.terminal_states != []:
                for terminal_state in self.terminal_states:
                    x, y = terminal_state
                    self.rewards[y][x] = StochasticReward(deterministic = True, mean = 0)
            else:
                # If no terminal states, set the reward of the bottom-right cell to 0
                self.rewards[self.height - 1][self.width - 1] = StochasticReward(deterministic = True, mean = 0)
                self.terminal_states = [(self.height - 1, self.width - 1)]
        else:
            if isinstance(rewards, (list, np.ndarray)) and all(isinstance(row, (list, np.ndarray)) and all(isinstance(val, (float, int)) for val in row) for row in rewards):
                # If rewards is a 2D list of numbers, use them as deterministic rewards
                self.rewards = [[StochasticReward(value, deterministic=True, mean=value) for value in row] for row in rewards]
            else:
                self.rewards = rewards
        self.expected_rewards = np.array([[reward.expected_value() for reward in row] for row in self.rewards])
        assert self.expected_rewards.shape == (self.height, self.width), "Rewards array size should match grid dimensions"

        # State initialization (ensure self.reset() exists)
        self.encoded_state = self.reset()
        self.state = self.decode_state(self.encoded_state)

        # Transition sequence
        self.transitions = []

        # Construct weights and feature vector
        self.task = task if task is not None else "Varying Blocks, Same Values"
        if task == "Same Blocks, Varying Values":
            self.distinct_rewards = distinct_rewards
        self.weights = self.get_weights()
        self.n_weights = len(self.weights)

        # Barrier set
        self.barrier_set = set()
        for state in range(self.n_states):
            if self.get_expected_reward(self.decode_state(state), 0, self.decode_state(state)) < 0:
                self.barrier_set.add(state)
        
        # Danger set
        self.danger_sets = []
        for element in self.barrier_set:
            danger_set = {element}
            self.danger_sets.append(danger_set)


    # ------------- State Handling -------------
    def reset(self) -> int:
        self.transitions = []  # Reset transition sequence
        if self.initial_state is not None:
            # Reset to the initial state
            self.state = self.initial_state
        else:
            # Reset to a random starting position not in terminal states
            while True:
                x = np.random.randint(self.width)
                y = np.random.randint(self.height)
                if (x, y) not in self.terminal_states:
                    break
            self.state = (x,y)
        self.encoded_state = self.encode_state(self.state)
        return self.encoded_state

    def is_terminal_state(self, state: Tuple[int, int]) -> bool:
        # Check if given state is terminal based on self.rewards
        return state in self.terminal_states

    def get_reward(self, state, action, next_state) -> float:
        # Get reward for specific position in the grid
        return self.rewards[next_state[1]][next_state[0]]
    
    def get_sampled_reward(self, state, action, next_state) -> float:
        # Get sampled reward for a given state-action pair
        return self.rewards[next_state[1]][next_state[0]].sample()[0]
    
    def get_expected_reward(self, state, action, next_state) -> float:
        # Get expected reward for a given state-action pair
        return self.rewards[next_state[1]][next_state[0]].expected_value()
    
    def get_expected_reward_squared(self, state, action, next_state) -> float:
        # Get expected squared reward for a given state-action pair
        return self.rewards[next_state[1]][next_state[0]].expected_value_squared()
    
    def get_reward_variance(self, state, action, next_state) -> float:
        # Get reward variance for a given state-action pair
        return self.rewards[next_state[1]][next_state[0]].variance()
    
    def encode_state(self, state: Tuple[int, int]) -> int:
        # Encode the state as a single integer
        x, y = state
        return x * self.width + y
    
    def decode_state(self, encoded_state: int) -> Tuple[int, int]:
        # Decode the integer state representation
        x = encoded_state // self.width
        y = encoded_state % self.width
        return x, y
    

    # ------------- Successor Feature Handling -------------
    def get_weights(self) -> List[float]:
        """
        Retrieves the weight vector for the current task.

        Returns:
        - List[float]: A list of weights. For the task "Same Blocks, Varying Values", this list represents the unique rewards encountered, padded with zeros if necessary to match a predefined number of distinct rewards. For other tasks, it generates a vector of expected rewards for each state.
        """

        if self.task == "Same Blocks, Varying Values":
            weights = np.unique(self.rewards)
            if len(weights) < self.distinct_rewards:
                # append zeros at the beginning of the array
                weights = np.append(np.zeros(self.distinct_rewards - len(weights)), weights)
        else:
            weights = np.zeros(self.expected_rewards.shape[0] * self.expected_rewards.shape[1])
            for state in range(self.n_states):
                x, y = self.decode_state(state)
                weights[state] = self.expected_rewards[y][x]

        return weights

    def get_feature_vector(self, state, action, next_state) -> np.ndarray:
        """
        Computes the feature vector for a given state-action-next state transition.

        Parameters:
        - state: The current state from which the action is taken.
        - action: The action taken by the agent.
        - next_state: The state reached after taking the action.

        Returns:
        - np.ndarray: A binary vector indicating the occurrence of specific features or rewards as a result of the state-action-next state transition.
        """

        if self.task == "Same Blocks, Varying Values":
            reward = self.get_reward(state, action, next_state)
            feature_vector = (self.weights == reward).astype(int)
        else: 
            feature_vector = np.zeros(self.n_weights)
            feature_vector[self.encode_state(next_state)] = 1
        return feature_vector


    # ------------- Agent Movement -------------
    def move_agent(self, action: int) -> Tuple[int, int]:
        """
        Moves the agent to a new position in the grid based on the specified action, considering barriers.

        Parameters:
        - action (int): The action to be taken by the agent. Actions are encoded as integers representing directions (e.g., UP, DOWN, LEFT, RIGHT).

        Returns:
        - Tuple[int, int]: The new position of the agent in the grid as (x, y) coordinates.

        """
        
        x, y = self.state  # Current position

        if action == self.UP and y < self.height - 1 and (x, y + 1) not in self.barrier_states:
            return x, y + 1  # Move Up
        elif action == self.DOWN and y > 0 and (x, y - 1) not in self.barrier_states:
            return x, y - 1  # Move Down
        elif action == self.LEFT and x > 0 and (x - 1, y) not in self.barrier_states:
            return x - 1, y  # Move Left
        elif action == self.RIGHT and x < self.width - 1 and (x + 1, y) not in self.barrier_states:
            return x + 1, y  # Move Right
        else:
            return x, y  # No move or blocked by barrier, stay in current positio
        
    def set_action_probabilities(self, action: int) -> List[float]:
        """
        Calculate and return the transition probabilities for each action based on the current action.
        
        Parameters:
        - action (int): The current action taken by the agent. Actions are encoded as integers.
        
        Returns:
        - List[float]: A list of probabilities corresponding to each action. The list's length is equal to the number of actions, and each element represents the probability of transitioning to the corresponding action from the current action.
        
        """
        
        probabilities = [0.0] * self.n_actions

        # Define neighboring actions based on the current action
        right_action = self.actions[(action + 1) % self.n_actions]
        left_action = self.actions[(action - 1) % self.n_actions]
        down_action = self.actions[(action + 2) % self.n_actions]

        # Assign transition probabilities to actions
        probabilities[action] = self.transition_probabilities[0]
        probabilities[right_action] = self.transition_probabilities[1]
        probabilities[left_action] = self.transition_probabilities[2]
        probabilities[down_action] = self.transition_probabilities[3]

        return probabilities

    def step(self, input_action: int) -> Tuple[int, float, bool, dict]:
        """
        Execute a step in the GridWorld environment based on the input action.

        Parameters:
        - input_action (int): The action chosen by the agent to perform in the current state. The actions are encoded as integers.

        Returns:
        - Tuple[int, float, bool, dict]: A tuple containing the new encoded state, reward received, a boolean indicating if the episode has finished, and an empty dictionary that can be used for additional information in the future.
        
         """

        current_state = self.state

        probabilities = self.set_action_probabilities(input_action)

        # Define the numbers and their probabilities
        actions = list(range(self.n_actions))

        # Sample an action with the specified probabilities
        sampled_action = np.random.choice(actions, p=probabilities)

        # Get the new state after the agent's movement based on the sampled action
        new_state = self.move_agent(sampled_action)

        # Calculate the reward of the new state
        reward = self.get_sampled_reward(current_state, sampled_action, new_state)

        # Check if the episode is done (new state is a terminal state)
        done = self.is_terminal_state(new_state)

        # Update the state
        self.state = new_state
        self.encoded_state = self.encode_state(self.state)

        # Record the transition
        self.transitions.append((current_state, sampled_action, reward, new_state, done))

        return self.encoded_state, reward, done, {}

    
    # ------------- Visualization -------------
    def arrow_map(self, action):
        """
        Converts action integers into arrow symbols.

        Parameters:
        - action (int): Integer representing the action.

        Returns:
        - str: Corresponding arrow symbol.
        """
        if action == self.UP:
            return '↓' # since the y-axis is inverted
        elif action == self.DOWN:
            return '↑' # since the y-axis is inverted
        elif action == self.LEFT:
            return '←'
        elif action == self.RIGHT:
            return '→'
        else:
            return '⊗'  # Symbol for an unknown or invalid action (optional)
    
    def render(self):
        """
        Render a heatmap depicting arrow transitions based on the agent's simulation.
        """
        # Generate arrow transitions based on the agent's actions during simulation
        arrow_transitions = [self.arrow_map(action) for (_, action, _, _, _) in self.transitions]
        
        # Extract locations from the simulation states
        locations = [(x, y) for ((x, y), _, _, _, _) in self.transitions]

        # Plot heatmap using rewards and arrow transitions
        heat_map(self.expected_rewards, arrow_transitions, locations)

    def plot_policy(self, pi, values=None, title=None):
        """
        Plot the policy values for each state.

        Parameters:
        pi (np.ndarray): Policy values for each state.
        values (2D array): Values to be plotted as a heatmap (default is self.rewards).
        title (str): Title of the plot (default is 'Heat Map of Policy').
        """
        if values is None:
            values = self.expected_rewards

        if title is None:
            title = "Heat Map of Policy"

        # Create arrows depicting policy values for non-terminal states
        index_terminal_states = [self.encode_state(state) for state in self.terminal_states]
        arrow_policy = []
        locations = []
        for i in range(self.n_states):
            if i not in index_terminal_states:
                locations.append(self.decode_state(i))
                pi_state = pi[i]
                arrow_state = []
                for action in range(self.n_actions):
                    if pi_state[action] != 0:
                        arrow_state.append(self.arrow_map(action))
                arrow_policy.append(arrow_state)

        # Plot heatmap using policy values and arrow representations
        heat_map(values, arrow_policy, locations, title=title)

    def plot_trajectory(self, pi, title = None):
        assert self.initial_state is not None, "Initial state not specified"
        if title is None:
            title = "Trajectory"
        self.reset()
        self.state = self.initial_state
        locations = [self.state]
        actions = []
        while self.state not in self.terminal_states:
            action = pi[self.encode_state(self.state)]
            self.state = self.move_agent(action)
            actions.append(action)
            locations.append(self.state)
        arrows = [self.arrow_map(action) for action in actions]
        heat_map(self.expected_rewards, arrows, locations, title=title)

    def __str__(self):
        # Print the GridWorld object
        return f"GridWorld(height={self.height}, width={self.width}, rewards={self.expected_rewards}, terminal_states={self.terminal_states}, barrier_states={self.barrier_states})"


    # ------------- Close ------------- 
    def close(self):
        pass

        


