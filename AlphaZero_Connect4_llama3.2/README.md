# Llama-Based AlphaZero for Connect4

This repository contains a Python implementation of the AlphaZero algorithm, adapted to play Connect4 using a Llama language model as its policy and value network. The code leverages PyTorch and the Hugging Face Transformers library.

## Overview

This project demonstrates how a large language model (LLM), specifically a Llama model, can be used within a reinforcement learning framework like AlphaZero to learn to play a game. Here's a breakdown of the key components:

*   **Connect4 Game Engine:** A simple and efficient implementation of the Connect4 game logic.
*   **LlamaConnect4 Model:**  A PyTorch module that wraps a Llama model. It takes a board state as input (converted to text) and outputs:
    *   A value score representing the desirability of the current state.
    *   A policy representing the probability distribution over possible actions (columns).
*   **Monte Carlo Tree Search (MCTS):** A search algorithm that uses the neural network to guide exploration and select the best move.
*   **AlphaZero Training:** A self-play training loop that:
    *   Generates game data through self-play using MCTS.
    *   Stores game states and results in memory.
    *   Trains the Llama-based policy/value network via gradient descent.
*   **Evaluation:** A class that assesses the policy network's accuracy using simple win/block examples.
*   **Play Agents:**
    *   `play_pth.py` can load and play an agent using only the saved `.pth` file containing trained weights.
    *   `play_llama.py` can load and play an agent using the saved Llama model directory, including tokenizer and model configuration files.
*   **Training Script:** `train1.py` which trains the model using self-play and MCTS.

## Files

*   `train1.py`: The main training script for the AlphaZero agent.
*   `play_pth.py`: A script to play Connect4 against the trained agent, loading only `.pth` file.
*   `play_llama.py`:  A script to play Connect4 against the trained agent, loading a full saved llama model directory.

## Setup

*   You need access to a Llama model. The code is currently configured to load a local Llama model from a specified directory, i.e. `"/Users/rbutler/Desktop/WORK_FT/LLAMA_32_1B_I/"`
*   The `model_path` in `LlamaConnect4` in each of the Python files will have to be updated to match your model directory.
*   You will also have to have the saved directory of an already fine-tuned model, in order for loading a whole llama model directory to work (`play_llama.py`)

## Usage

### 1. Training

To train a new model from scratch, run `train1.py`:

```bash
python train1.py
```

This script will:
* Train the policy/value network via self-play.
* Evaluate the policy network's accuracy on simple win/block scenarios after each epoch of training.
* Save the trained model weights in PyTorch format at `./alphazero-network-weights.pth`.
* Also saves the full Llama model (with tokenizer) to a directory: `./FT_LLAMA_32_1B_I/` .

Training parameters are configured within the `config_dict` in the `train1.py` file. Modify these to adjust training behavior.

### 2. Playing

To play against the trained agent (using `.pth` weights) run:

```bash
python play_pth.py
```

To play against the trained agent (using full saved llama model directory) run:

```bash
python play_llama.py
```

The program will prompt you to enter your move using column numbers (0-6).  The number of iterations available to the mcts search (`mcts_max_search_iter`) will greatly affect the length of time for choosing a move, e.g. setting it to 0 will result in near instantaneous moves, however it will be far weaker.  A value of 200 or higher will result in much better gameplay, but might take 2-3 minutes per move on CPU.

## Key Configurations

The `config_dict` at the beginning of each file controls the behaviour of both the training and playing scripts.  Here are some of the more important configurations:

*   `device`: Specifies the device to use for training (e.g., `cuda` if available or `cpu`).
*   `learning_rate`:  Adam learning rate.
*   `training_epochs`: How many full training epochs to do.
*   `games_per_epoch`: How many self-played games per epoch.
*   `minibatch_size`: Size of each minibatch used in learning update.
*   `mcts_start_search_iter`: Number of Monte Carlo tree search iterations initially.
*   `mcts_max_search_iter`:  Maximum number of MCTS iterations during training.  Note that this can be increased further in the playing scripts.
*   `temperature`: Selection temperature. A greater temperature is a more uniform distribution
*   `dirichlet_eps`: Weight of dirichlet noise.

## Notes

*   This implementation uses a prompt-based approach for using Llama as value/policy network. This might not be as efficient as a more direct, token-based approach.
*   The performance of the agent depends greatly on the number of MCTS iterations.  Increasing search iterations will improve play but will come at a cost of slower decision making.
*   This model is trained using self-play.  To improve the gameplay further, fine-tuning the Llama model using data generated from better Connect4 agents may yield better results.
*   The `Evaluator` class provides an accuracy measure on simple win/block scenarios, but won't be very useful if the neural network isn't making much of a useful move.
*   This code uses a 1B parameter version of Llama, and requires a lot of memory.

## Contributing

Contributions to this project are welcome. Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License.
