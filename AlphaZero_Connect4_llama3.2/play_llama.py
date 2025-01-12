import torch
import sys, os, time
from transformers import AutoTokenizer, LlamaForCausalLM
from torch import nn
import torch.nn.functional as F
import math
import random
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from IPython.display import HTML
from base64 import b64encode


config_dict = {
    'device': torch.device('cuda') if torch.cuda.is_available() else 'cpu',
    'n_filters': 128,              # Number of convolutional filters used in residual blocks
    'n_res_blocks': 8,             # Number of residual blocks used in network
    'exploration_constant': 2,     # Exploration constant used in PUCT calculation
    'temperature': 1.25,           # Selection temperature. A greater temperature is a more uniform distribution
    'dirichlet_alpha': 1.,         # Alpha parameter for Dirichlet noise. Larger values mean more uniform noise
    'dirichlet_eps': 0.25,         # Weight of dirichlet noise
    'learning_rate': 0.001,        # Adam learning rate
    'training_epochs': 2,    ###  # How many full training epochs
    'games_per_epoch': 2,    ### # How many self-played games per epoch
    'minibatch_size': 4,   ###   # Size of each minibatch used in learning update 
    'n_minibatches': 2,      ###   # How many minibatches to accumulate per learning step
    'mcts_start_search_iter': 10, ###  # Number of Monte Carlo tree search iterations initially
    'mcts_max_search_iter': 100, ###   # Maximum number of MCTS iterations
    'mcts_search_increment': 1,    # After each epoch, how much should search iterations be increased by
    }

# Convert to a struct esque object
class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

config = Config(config_dict)

class Connect4:
    "Connect 4 game engine, containing methods for game-related tasks."
    def __init__(self):
        self.rows = 6
        self.cols = 7

    def get_next_state(self, state, action, to_play=1):
        "Play an action in a given state and return the resulting board."
        # Pre-condition checks
        assert self.evaluate(state) == 0
        assert np.sum(abs(state)) != self.rows * self.cols
        assert action in self.get_valid_actions(state)
        
        # Identify next empty row in column
        row = np.where(state[:, action] == 0)[0][-1]
        
        # Apply action
        new_state = state.copy()
        new_state[row, action] = to_play
        return new_state

    def get_valid_actions(self, state):
        "Return a numpy array containing the indices of valid actions."
        # If game over, no valid moves
        if self.evaluate(state) != 0:
            return np.array([])
        
        # Identify valid columns to play
        cols = np.sum(np.abs(state), axis=0)
        return np.where((cols // self.rows) == 0)[0]

    def evaluate(self, state):
        "Evaluate the current position. Returns 1 for player 1 win, -1 for player 2 and 0 otherwise."
        # Kernels for checking win conditions
        kernel = np.ones((1, 4), dtype=int)
        
        # Horizontal and vertical checks
        horizontal_check = convolve2d(state, kernel, mode='valid')
        vertical_check = convolve2d(state, kernel.T, mode='valid')

        # Diagonal checks
        diagonal_kernel = np.eye(4, dtype=int)
        main_diagonal_check = convolve2d(state, diagonal_kernel, mode='valid')
        anti_diagonal_check = convolve2d(state, np.fliplr(diagonal_kernel), mode='valid')
        
        # Check for winner
        if any(cond.any() for cond in [horizontal_check == 4, vertical_check == 4, main_diagonal_check == 4, anti_diagonal_check == 4]):
            return 1
        elif any(cond.any() for cond in [horizontal_check == -4, vertical_check == -4, main_diagonal_check == -4, anti_diagonal_check == -4]):
            return -1

        # No winner
        return 0  

    def step(self, state, action, to_play=1):
        "Play an action in a given state. Return the next_state, reward and done flag."
        # Get new state and reward
        next_state = self.get_next_state(state, action, to_play)
        reward = self.evaluate(next_state)
        
        # Check for game termination
        done = True if reward != 0 or np.sum(abs(next_state)) >= (self.rows * self.cols - 1) else False
        return next_state, reward, done

    def encode_state(self, state):
        "Convert state to tensor with 3 channels."
        encoded_state = np.stack((state == 1, state == 0, state == -1)).astype(np.float32)
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        return encoded_state

    def reset(self):
        "Reset the board."
        return np.zeros([self.rows, self.cols], dtype=np.int8)

class LlamaConnect4(nn.Module):
    "Llama-based model for Connect4"
    def __init__(self, game, config):
        super().__init__()
        self.game = game
        self.config = config  # Store config as instance variable
        
        try:
            model_path = "/Users/rbutler/Desktop/WORK_FT/FT_LLAMA_32_1B_I/"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = LlamaForCausalLM.from_pretrained(model_path)
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id  # Add this line
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure you have access to the Llama-2 model")
            sys.exit(-1)

    def forward(self, x):
        values = []
        policies = []
        
        for board in x:
            # Convert board state to text prompt
            prompt = f"Given this Connect4 board state:\n{board}\nPredict the best move and evaluate the position."
            
            # Generate response from Llama model
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.config.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                return_dict_in_generate=True,
                output_scores=True,
                return_legacy_cache=True
            )

            # Add this inside forward() right after outputs = self.model.generate():
            # Decode and print the full response
            full_response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

            # Get logits from first prediction step
            logits = outputs.scores[0][0]  # Shape should be [vocab_size]
            
            # Calculate value (between -1 and 1) from the full logits
            value = torch.tanh(logits.mean())
            values.append(value)
            
            # For policy, take first 7 logits (one per column)
            # If needed, you might want to map specific tokens to columns instead
            policy_logits = logits[:self.game.cols]
            policy = F.softmax(policy_logits, dim=0)
            policies.append(policy)
        
        # Stack into batched tensors
        value_tensor = torch.stack(values)
        policy_tensor = torch.stack(policies)
        
        return value_tensor, policy_tensor

class MCTS:
    def __init__(self, network, game, config):
        "Initialize Monte Carlo Tree Search with a given neural network, game instance, and configuration."
        self.network = network
        self.game = game
        self.config = config

    def search(self, state, total_iterations, temperature=None):
        "Performs a search for the desired number of iterations, returns an action and the tree root."
        # Create the root
        root = Node(None, state, 1, self.game, self.config)

        # Expand the root, adding noise to each action
        # Get valid actions
        valid_actions = self.game.get_valid_actions(state)
        state_tensor = torch.tensor(self.game.encode_state(state), dtype=torch.float).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            self.network.eval()
            value, logits = self.network(state_tensor)

        # Get action probabilities
        action_probs = F.softmax(logits.view(self.game.cols), dim=0).cpu().numpy()

        # Print initial state
        print("\nDEBUG Initial Root Evaluation:")
        print("  Valid actions:", valid_actions)
        print("  Initial action probabilities:", action_probs)

        # Calculate and add dirichlet noise
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * self.game.cols)
        action_probs = ((1 - self.config.dirichlet_eps) * action_probs) + self.config.dirichlet_eps * noise

        # Mask unavailable actions
        mask = np.full(self.game.cols, False)
        mask[valid_actions] = True
        action_probs = action_probs[mask]

        # Ensure action_probs are valid probabilities
        if np.sum(action_probs) == 0 or np.any(np.isnan(action_probs)):
            action_probs = np.ones_like(action_probs) / len(action_probs)
        else:
            action_probs = action_probs / np.sum(action_probs)

        # Create children with validated probabilities
        for action, prob in zip(valid_actions, action_probs):
            child_state = -self.game.get_next_state(state, action)
            root.children[action] = Node(root, child_state, -1, self.game, self.config)
            root.children[action].prob = float(prob)

        # Since we're not backpropagating, manually increase visits
        root.n_visits = 1
        root.total_score = value.item()

        # Begin search
        print("\nStarting MCTS with", total_iterations, "iterations")
        for i in range(total_iterations):
            current_node = root
            search_path = []

            # Phase 1: Selection
            while not current_node.is_leaf():
                current_node = current_node.select_child()
                search_path.append(current_node)

            # Phase 2: Expansion
            if not current_node.is_terminal():
                current_node.expand()
                state_tensor = torch.tensor(self.game.encode_state(current_node.state), dtype=torch.float).unsqueeze(0).to(self.config.device)
                with torch.no_grad():
                    self.network.eval()
                    value, logits = self.network(state_tensor)
                    value = value.item()
            else:
                value = self.game.evaluate(current_node.state)

            # Phase 3: Backpropagation
            current_node.backpropagate(value)

            # Print progress every few iterations
            if (i + 1) % 5 == 0:
                print(f"Completed iteration {i + 1}/{total_iterations}")
        
        print("\nMCTS Search completed")
        
        # Select action with specified temperature
        if temperature == None:
            temperature = self.config.temperature
        return self.select_action(root, temperature), root

    def select_action(self, root, temperature=None):
        "Select an action from the root based on visit counts, adjusted by temperature, 0 temp for greedy."
        if temperature == None:
            temperature = self.config.temperature
        action_counts = {key: val.n_visits for key, val in root.children.items()}
        if temperature == 0:
            return max(action_counts, key=action_counts.get)
        elif temperature == np.inf:
            return np.random.choice(list(action_counts.keys()))
        else:
            distribution = np.array([*action_counts.values()]) ** (1 / temperature)
            return np.random.choice([*action_counts.keys()], p=distribution/sum(distribution))

class Node:
    def __init__(self, parent, state, to_play, game, config):
        "Represents a node in the MCTS, holding the game state and statistics for MCTS to operate."
        self.parent = parent
        self.state = state
        self.to_play = to_play
        self.config = config
        self.game = game

        self.prob = 0
        self.children = {}
        self.n_visits = 0
        self.total_score = 0

    def expand(self):
        "Create child nodes for all valid actions. If state is terminal, evaluate and set the node's value."
        # Get valid actions
        valid_actions = self.game.get_valid_actions(self.state)

        # If there are no valid actions, state is terminal, so get value using game instance
        if len(valid_actions) == 0:
            self.total_score = self.game.evaluate(self.state)
            return

        # Create a child for each possible action
        for action in valid_actions:    # Removed the zip
            # Make move, then flip board to perspective of next player
            child_state = -self.game.get_next_state(self.state, action)
            self.children[action] = Node(self, child_state, -self.to_play, self.game, self.config)

    def select_child(self):
        "Select the child node with the highest PUCT score."
        best_puct = -np.inf
        best_child = None
        for action, child in self.children.items():
            puct = self.calculate_puct(child)
            if puct > best_puct:
                best_puct = puct
                best_child = child
        return best_child

    def calculate_puct(self, child):
        "Calculate the PUCT score for a given child node."
        exploitation_term = 1 - (child.get_value() + 1) / 2
        exploration_term = child.prob * math.sqrt(self.n_visits) / (child.n_visits + 1)
        return exploitation_term + self.config.exploration_constant * exploration_term

    def backpropagate(self, value):
        "Update the current node and its ancestors with the given value."
        self.total_score += value
        self.n_visits += 1
        if self.parent is not None:
            # Backpropagate the negative value so it switches each level
            self.parent.backpropagate(-value)

    def is_leaf(self):
        "Check if the node is a leaf (no children)."
        return len(self.children) == 0

    def is_terminal(self):
        "Check if the node represents a terminal state."
        return (self.n_visits != 0) and (len(self.children) == 0)

    def get_value(self):
        "Calculate the average value of this node."
        if self.n_visits == 0:
            return 0
        return self.total_score / self.n_visits
    
    def __str__(self):
        "Return a string containing the node's relevant information for debugging purposes."
        return (f"State:\n{self.state}\nProb: {self.prob}\nTo play: {self.to_play}" +
                f"\nNumber of children: {len(self.children)}\nNumber of visits: {self.n_visits}" +
                f"\nTotal score: {self.total_score}")


class AlphaZero:
    def __init__(self, game, config, verbose=True):
        self.network = LlamaConnect4(game, config).to(config.device)
        self.mcts = MCTS(self.network, game, config)
        self.game = game
        self.config = config

        # Losses and optimizer
        self.loss_cross_entropy = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate, weight_decay=0.0001)

        # Pre-allocate memory on GPU
        state_shape = game.encode_state(game.reset()).shape
        self.max_memory = config.minibatch_size * config.n_minibatches
        self.state_memory = torch.zeros(self.max_memory, *state_shape).to(config.device)
        self.value_memory = torch.zeros(self.max_memory, 1).to(config.device)
        self.policy_memory = torch.zeros(self.max_memory, game.cols).to(config.device)
        self.current_memory_index = 0
        self.memory_full = False

        # MCTS search iterations
        self.search_iterations = config.mcts_start_search_iter
        
        # Logging
        self.verbose = verbose
        self.total_games = 0

    def train(self, training_epochs):
        "Train the AlphaZero agent for a specified number of training epochs."
        # For each training epoch
        for _ in range(training_epochs):

            # Play specified number of games
            for _ in range(self.config.games_per_epoch):
                print("   DBG START ONE SELF PLAY")
                stime = time.time()
                self.self_play()
                etime = time.time()
                print("   DBG DONE ONE SELF PLAY", etime-stime)
            
            # At the end of each epoch, increase the number of MCTS search iterations
            self.search_iterations = min(self.config.mcts_max_search_iter, self.search_iterations + self.config.mcts_search_increment)

    def self_play(self):
        "Perform one episode of self-play."
        state = self.game.reset()
        done = False
        while not done:
            # Search for a move
            print("   DOING MCTS SEARCH")
            stime = time.time()
            action, root = self.mcts.search(state, self.search_iterations)
            etime = time.time()
            print("   DONE MCTS SEARCH",etime-stime)

            # Value target is the value of the MCTS root node
            value = root.get_value()

            # Visit counts used to compute policy target
            visits = np.zeros(self.game.cols)
            for child_action, child in root.children.items():
                visits[child_action] = child.n_visits
            # Softmax so distribution sums to 1
            visits /= np.sum(visits)

            # Append state + value & policy targets to memory
            self.append_to_memory(state, value, visits)

            # If memory is full, perform a learning step
            if self.memory_full:
                print("     DOING SELF LEARN")
                stime = time.time()
                self.learn()
                etime = time.time()
                print("     DONE SELF LEARN", etime-stime)

            # Perform action in game
            state, _, done = self.game.step(state, action)

            # Flip the board
            state = -state

        # Increment total games played
        self.total_games += 1

        # Logging if verbose
        if self.verbose:
            print("\rTotal Games:", self.total_games, "Items in Memory:", self.current_memory_index, "Search Iterations:", self.search_iterations, " ", end="")

    def append_to_memory(self, state, value, visits):
        """
        Append state and MCTS results to memory buffers.
        Args:
            state (array-like): Current game state.
            value (float): MCTS value for the game state.
            visits (array-like): MCTS visit counts for available moves.
        """
        # Calculate the encoded states
        encoded_state = np.array(self.game.encode_state(state))
        encoded_state_augmented = np.array(self.game.encode_state(state[:, ::-1]))

        # Stack states and visits
        states_stack = np.stack((encoded_state, encoded_state_augmented), axis=0)
        visits_stack = np.stack((visits, visits[::-1]), axis=0)

        # Convert the stacks to tensors
        state_tensor = torch.tensor(states_stack, dtype=torch.float).to(self.config.device)
        visits_tensor = torch.tensor(visits_stack, dtype=torch.float).to(self.config.device)
        value_tensor = torch.tensor(np.array([value, value]), dtype=torch.float).to(self.config.device).unsqueeze(1)

        # Store in pre-allocated GPU memory
        self.state_memory[self.current_memory_index:self.current_memory_index + 2] = state_tensor
        self.value_memory[self.current_memory_index:self.current_memory_index + 2] = value_tensor
        self.policy_memory[self.current_memory_index:self.current_memory_index + 2] = visits_tensor

        # Increment index, handle overflow
        self.current_memory_index = (self.current_memory_index + 2) % self.max_memory

        # Set memory filled flag to True if memory is full
        if (self.current_memory_index == 0) or (self.current_memory_index == 1):
            self.memory_full = True


    def learn(self):
        "Update the neural network by extracting minibatches from memory and performing one step of optimization for each one."
        self.network.train()

        # Create a randomly shuffled list of batch indices
        batch_indices = np.arange(self.max_memory)
        np.random.shuffle(batch_indices)

        for batch_index in range(self.config.n_minibatches):
            # Get minibatch indices
            start = batch_index * self.config.minibatch_size
            end = start + self.config.minibatch_size
            mb_indices = batch_indices[start:end]

            # Slice memory tensors
            mb_states = self.state_memory[mb_indices]
            mb_value_targets = self.value_memory[mb_indices]
            mb_policy_targets = self.policy_memory[mb_indices]

            # Network predictions
            value_preds, policy_logits = self.network(mb_states)

            # Loss calculation
            policy_loss = self.loss_cross_entropy(policy_logits, mb_policy_targets)
            value_loss = self.loss_mse(value_preds.view(-1), mb_value_targets.view(-1))
            loss = policy_loss + value_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory_full = False
        self.network.eval()


class Evaluator:
    "Class to evaluate the policy network's performance on simple moves."
    def __init__(self, alphazero, num_examples=500, verbose=True):
        self.network = alphazero.network
        self.game = alphazero.game
        self.config = alphazero.config
        self.accuracies = []
        self.num_examples = num_examples
        self.verbose = verbose

        # Generate and prepare example states and actions for evaluation
        self.generate_examples()

    def select_action(self, state):
        "Select an action based on the given state, will choose a winning or blocking moves."
        valid_actions = self.game.get_valid_actions(state)
        
        # Check for a winning move
        for action in valid_actions:
            next_state, reward, _ = self.game.step(state, action)
            if reward == 1:
                return action

        # Check for a blocking move
        flipped_state = -state
        for action in valid_actions:
            next_state, reward, _ = self.game.step(flipped_state, action)
            if reward == 1:
                return action

        # Default to random action if no winning or blocking move
        return random.choice(valid_actions)

    def generate_examples(self):
        "Generate and prepare example states and actions for evaluation."
        winning_examples = self.generate_examples_for_condition('win')
        blocking_examples = self.generate_examples_for_condition('block')

        # Prepare states and actions for evaluation
        winning_example_states, winning_example_actions = zip(*winning_examples)
        blocking_example_states, blocking_example_actions = zip(*blocking_examples)

        target_states = np.concatenate([winning_example_states, blocking_example_states], axis=0)
        target_actions = np.concatenate([winning_example_actions, blocking_example_actions], axis=0)

        encoded_states = [self.game.encode_state(state) for state in target_states]
        self.X_target = torch.tensor(np.stack(encoded_states, axis=0), dtype=torch.float).to(self.config.device)
        self.y_target = torch.tensor(target_actions, dtype=torch.long).to(self.config.device)

    def generate_examples_for_condition(self, condition):
        "Generate examples based on either 'win' or 'block' conditions."
        examples = []
        while len(examples) < self.num_examples:
            state = self.game.reset()
            while True:
                action = self.select_action(state)
                next_state, reward, done = self.game.step(state, action, to_play=1)
                
                if condition == 'win' and reward == 1:
                    examples.append((state, action))
                    break
                
                if done:
                    break
                
                state = next_state

                # Flipping the board for opponent's perspective
                action = self.select_action(-state)
                next_state, reward, done = self.game.step(state, action, to_play=-1)
                
                if condition == 'block' and reward == -1:
                    examples.append((-state, action))
                    break
                
                if done:
                    break
                
                state = next_state
        return examples

    def evaluate(self):
        "Evaluate the policy network's accuracy and append it to self.accuracies."
        print(f"\nEvaluating {len(self.X_target)} examples...")
        batch_size = 50
        all_pred_actions = []
        
        with torch.no_grad():
            self.network.eval()
            for i in range(0, len(self.X_target), batch_size):
                batch = self.X_target[i:i+batch_size]
                _, logits = self.network(batch)
                pred_actions = logits.argmax(dim=1)
                all_pred_actions.append(pred_actions)
                print(f"Progress: {min(i+batch_size, len(self.X_target))}/{len(self.X_target)} examples processed")
            
            all_pred_actions = torch.cat(all_pred_actions)
            accuracy = (all_pred_actions == self.y_target).float().mean().item()
        
        self.accuracies.append(accuracy)
        if self.verbose:
            print(f"\nEvaluation Accuracy: {100 * accuracy:.1f}%")

print("DBG1")
game = Connect4()
print("DBG2")
alphazero = AlphaZero(game, config)
print("DBG3")

# file_path = "./alphazero-network-weights.pth"
# pre_trained_weights = torch.load(file_path, map_location=config.device, weights_only=True)
# 
# alphazero.network.load_state_dict(pre_trained_weights)

class AlphaZeroAgent:
    def __init__(self, alphazero):
        self.alphazero = alphazero
        self.alphazero.network.eval()
        
        # Remove noise from move calculations
        self.alphazero.config.dirichlet_eps = 0

    def select_action(self, state, search_iterations=200):
        stime = time.time()
        state_tensor = torch.tensor(self.alphazero.game.encode_state(state), dtype=torch.float).to(self.alphazero.config.device)
        
        # Get action without using search
        print("SELECTING ACTION; search_iterations ", search_iterations)
        if search_iterations == 0:
            with torch.no_grad():
                _, logits = self.alphazero.network(state_tensor.unsqueeze(0))

            # Get action probs and mask for valid actions
            action_probs = F.softmax(logits.view(-1), dim=0).cpu().numpy()
            valid_actions = self.alphazero.game.get_valid_actions(state)
            valid_action_probs = action_probs[valid_actions]
            best_action = valid_actions[np.argmax(valid_action_probs)]
            etime = time.time()
            print("QUICK BEST ACTION", best_action, "TIME", etime-stime)
            return best_action
        # Else use MCTS 
        else:
            action, _ = self.alphazero.mcts.search(state, search_iterations)
            etime = time.time()
            print("LONG CHOICE ACTION", action, "TIME", etime-stime)
            return action

agent = AlphaZeroAgent(alphazero)

# set to 1 for AlphaZero to play first
turn = 0

# reset the game
state = Connect4().reset()
done = False

# play loop
while not done:
    print("\nCurrent Board:")
    print(state)
    print("\nColumns are numbered 0-6 from left to right")

    if turn == 0:
        while True:
            try:
                print("\nHuman to move.")
                action = int(input("Enter column number (0-6): ").strip())
                if action >= 0 and action <= 6 and action in game.get_valid_actions(state):
                    break
                print("Invalid move! Please choose an available column (0-6)")
            except ValueError:
                print("Please enter a valid number between 0 and 6")
    else:
        print("\nAlphaZero is thinking...")
        action = agent.select_action(state, config.mcts_max_search_iter)  ###### RMB 0 up to 200)
        print(f"AlphaZero chose column {action}")

    next_state, reward, done = game.step(state, action)

    print("\nBoard After Move:")
    print(next_state)

    if done:
        if reward != 0:
            winner = "Human" if (reward == 1 and turn == 0) or (reward == -1 and turn == 1) else "AlphaZero"
            print(f"\nGame Over! {winner} wins!")
        else:
            print("\nGame Over! It's a draw!")
    else:
        state = -next_state
        turn = 1 - turn

