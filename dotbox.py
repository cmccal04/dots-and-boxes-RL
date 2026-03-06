import numpy as np
import random
import matplotlib.pyplot as plt

# ------------------------------
# Dots and Boxes Environment
# ------------------------------
class DotsAndBoxes:
    # Initializes the Dots and Boxes game environment
    def __init__(self, size=3):
        self.size = size
        self.reset()

    # Resets the game to the initial state
    def reset(self):
        # Initialize arrays for horizontal and vertical lines
        self.board_h = np.zeros((self.size + 1, self.size), dtype=int)
        self.board_v = np.zeros((self.size, self.size + 1), dtype=int)  
        self.boxes = np.zeros((self.size, self.size), dtype=int)

        # Generate the list of all possible moves (horizontal and vertical)
        self.available_moves = [("h", i, j) for i in range(self.size + 1) for j in range(self.size)] + \
                               [("v", i, j) for i in range(self.size) for j in range(self.size + 1)]
        
        # # Initialize scores for both players and set current player to player 1
        self.scores = {1: 0, 2: 0}
        self.current_player = 1

        # Return the current state of the game board
        return self.get_state()

    # Copy and return the current state of the game board
    def get_state(self):
        return (self.board_h.copy(), self.board_v.copy())

    # Copy and return the current available actions
    def available_actions(self):
        return self.available_moves.copy()

    # Executes a move and update the environment state
    def step(self, action):
        move_type, i, j = action
        # Illegal move penalty
        if action not in self.available_moves:
            return self.get_state(), -1, True

        # Update board
        if move_type == 'h':
            self.board_h[i, j] = 1
        else:
            self.board_v[i, j] = 1
        self.available_moves.remove(action)

        # After the move, check all boxes and assign points
        gained_box = False
        reward = 0
        for row in range(self.size):
            for col in range(self.size):
                if self.boxes[row, col] == 0 and self._is_box_complete(row, col):
                    self.boxes[row, col] = self.current_player
                    self.scores[self.current_player] += 1
                    reward += 1
                    gained_box = True

        # Switch players if no box was gained
        if not gained_box:
            self.current_player = 3 - self.current_player

        done = len(self.available_moves) == 0
        return self.get_state(), reward, done

    # Check if all four sides are drawn to create a box
    def _is_box_complete(self, row, col):
        top = self.board_h[row, col]
        bottom = self.board_h[row + 1, col]
        left = self.board_v[row, col]
        right = self.board_v[row, col + 1]
        return top and bottom and left and right

# -----------------
# Q-learning Agent
# -----------------
class QLearningAgent:
    def __init__(self, name, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.name = name
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}

    # Returns the current state - the key for the Q-table
    def get_state_key(self, board_h, board_v, current_player):
        # flatten boards to become key in Q-table
        return (tuple(board_h.flatten()), tuple(board_v.flatten()), current_player)

    # Implements action selection process
    def select_action(self, state_key, available_actions):
        # Select random action according to epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.choice(available_actions)

        # Evaluate Q-values for all available actions
        q_values = []
        for a in available_actions:
            key = (state_key, a)
            q_values.append(self.q_table.get(key, 0.0))

        # Find maximum Q-value, and handle ties with best_actions[]
        max_q = max(q_values)
        best_actions = []
        for i in range(len(available_actions)):
            action = available_actions[i]
            q_value = q_values[i]

            if q_value == max_q:
                best_actions.append(action)

        # Return any maximum Q-value
        return random.choice(best_actions)

    # Updates Q-table according to Q-learning update rule
    def update(self, state_key, action, reward, next_state_key, next_available_actions, done):
        # Set key and current Q-value (to be changed)
        key = (state_key, action)
        old_q = self.q_table.get(key, 0.0)

        if done:
            target = reward
        else:
            # Bellman equation
            next_qs = [self.q_table.get((next_state_key, a), 0.0) for a in next_available_actions]
            target = reward + self.gamma * max(next_qs) if next_qs else reward

        # Update Q-table using aplha (learning rate)
        self.q_table[key] = old_q + self.alpha * (target - old_q)

# -----------------
# Helper Functions
# -----------------

# Creates a copy of a given agent 
def copy_agent(agent):
    new_agent = QLearningAgent(name=agent.name)
    new_agent.q_table = dict(agent.q_table)
    new_agent.epsilon = agent.epsilon
    new_agent.alpha = agent.alpha
    new_agent.gamma = agent.gamma
    return new_agent

# Evaluates current state of an agent vs. opponent that selects actions randomly
def evaluate_against_random(agent, env, games=100):
    # Win counts
    wins = {1: 0, 2: 0}

    # Play through the games
    for _ in range(games):
        env.reset()
        done = False
        # Learning agent is player 1
        current_player = agent

        while not done:
            # Select actions using Q-learning for player 1. Select randomly for player 2
            if env.current_player == 1:
                state_key = current_player.get_state_key(env.board_h, env.board_v, env.current_player)
                available = env.available_actions()
                action = current_player.select_action(state_key, available)
            else:
                available = env.available_actions()
                action = random.choice(available)

            (board_h, board_v), reward, done = env.step(action)

        # Update win count
        if env.scores[1] > env.scores[2]:
            wins[1] += 1
        elif env.scores[1] < env.scores[2]:
            wins[2] += 1

    # Calculate final win rate
    win_rate = wins[1] / games if games > 0 else 0
    return win_rate

# ----------------------------
# Training - Direct Self-Play
# ----------------------------
def train_dsp(agent, env, episodes=10000, eval_interval=500):
    # Win counts and win rates trackers
    wins = {1: 0, 2: 0}
    win_rates_self_play = []
    win_rates_against_random = []

    score_margins = [] # Score margins

    # Main training loop for each episode
    for episode in range(1, episodes + 1):
        env.reset()
        done = False
        current_agent = agent
        other_agent = agent

        # Play a full game
        while not done:
            # Get current state and available actions
            state_key = current_agent.get_state_key(env.board_h, env.board_v, env.current_player)
            available_actions = env.available_actions()

            action = current_agent.select_action(state_key, available_actions) # Select action

            prev_score = env.scores[env.current_player] # Record score
            (board_h, board_v), reward, done = env.step(action) # Apply action
            gained_box = env.scores[env.current_player] > prev_score # Check if box was formed

            # Get next state
            next_state_key = current_agent.get_state_key(board_h, board_v, env.current_player)
            next_available = env.available_actions()

            # Determine winner and give reward
            if done:
                if env.scores[1] > env.scores[2]:
                    result = 1
                    reward = 10
                elif env.scores[1] < env.scores[2]:
                    result = 2
                    reward = -10
                wins[result] += 1

            # Update the agent and switch turn
            current_agent.update(state_key, action, reward, next_state_key, next_available, done)
            if not gained_box:
                current_agent, other_agent = other_agent, current_agent

        # Calculate score margin
        score_margin = env.scores[1] - env.scores[2]
        score_margins.append(score_margin)

        # Calculate win rate
        win_percent_self_play = wins[1] / episode if episode > 0 else 0
        win_rates_self_play.append(win_percent_self_play)

        # Evaulate versus random opponent
        if episode % eval_interval == 0:
            print(f"At Episode {episode}")
            random_win_rate = evaluate_against_random(agent, env)
            win_rates_against_random.append((episode, random_win_rate))

    return win_rates_self_play, win_rates_against_random, wins, score_margins

# ---------------------------------------------------
# Training - Fictional + Fictitious Self-Play Hybrid
# ---------------------------------------------------
def train_fsp(agent, env, episodes=5000, eval_interval=500):
    # Win counts and win rates trackers
    wins = {1: 0, 2: 0}
    win_rates_self_play = []
    win_rates_against_random = []

    # Copy in past agents
    past_agents = [copy_agent(agent)]

    score_margins = [] # Score margins

    # Main training loop for each episode
    for episode in range(1, episodes + 1):
        env.reset()
        done = False
        current_agent = agent
        opponent_agent = random.choice(past_agents) # Select a random past agent

        # Play a full game
        while not done:
            if env.current_player == 1:
                acting_agent = current_agent
            else:
                acting_agent = opponent_agent

            # Get current state and available actions
            state_key = acting_agent.get_state_key(env.board_h, env.board_v, env.current_player)
            available_actions = env.available_actions()

            action = acting_agent.select_action(state_key, available_actions) # Select action

            prev_score = env.scores[env.current_player] # Record score
            (board_h, board_v), reward, done = env.step(action) # Apply action
            gained_box = env.scores[env.current_player] > prev_score # Check if box is formed

            # Get next state
            next_state_key = acting_agent.get_state_key(board_h, board_v, env.current_player)
            next_available = env.available_actions()

            # Determine winner and give reward
            if done:
                if env.scores[1] > env.scores[2]:
                    result = 1
                    reward = 10
                elif env.scores[1] < env.scores[2]:
                    result = 2
                    reward = -10
                wins[result] += 1

            # Only update Q-table for learning agent (not opponent)
            if acting_agent == current_agent:
                acting_agent.update(state_key, action, reward, next_state_key, next_available, done)

            # Switch turns
            if not gained_box:
                if env.current_player == 1:
                    current_agent, opponent_agent = opponent_agent, current_agent

        # Calculate score margin
        score_margin = env.scores[1] - env.scores[2]
        score_margins.append(score_margin)

        # Calculate win rate
        win_percent_self_play = wins[1] / episode if episode > 0 else 0
        win_rates_self_play.append(win_percent_self_play)

        # Evaulate versus random opponent
        if episode % eval_interval == 0:
            print(f"At Episode {episode}")
            random_win_rate = evaluate_against_random(agent, env)
            win_rates_against_random.append((episode, random_win_rate))
            past_agents.append(copy_agent(agent))

    return win_rates_self_play, win_rates_against_random, wins, score_margins

# ---------------------------
# Visualization - by ChatGPT
# ---------------------------
def visualize_results(win_rates_self_play, win_rates_against_random, wins, score_margins):

    # --- Plot Win Rates ---
    plt.figure(figsize=(10, 5))
    plt.plot(win_rates_self_play, label='Self-Play Win %')
    if win_rates_against_random:
        episodes, random_rates = zip(*win_rates_against_random)
        plt.plot(episodes, random_rates, 'o-', label='Win % vs Random', markersize=5)
    plt.xlabel('Episode')
    plt.ylabel('Winning %')
    plt.title('Learning Curve (Self-Play and vs Random)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Compute Final Statistics ---
    final_self_play_win_rate = win_rates_self_play[-1] * 100  # %
    final_eval_win_rate = win_rates_against_random[-1][1] * 100 if win_rates_against_random else 0  # %
    avg_score_margin = np.mean(score_margins)
    total_wins_player1 = wins[1]
    total_wins_player2 = wins[2]

    # --- Prepare Summary Data ---
    data = [
        ["Final Self-Play Win Rate (%)", f"{final_self_play_win_rate:.2f}"],
        ["Final Win Rate vs Random (%)", f"{final_eval_win_rate:.2f}"],
        ["Average Score Margin", f"{avg_score_margin:.2f}"],
        ["Total Wins (Player 1)", f"{total_wins_player1}"],
        ["Total Wins (Player 2)", f"{total_wins_player2}"]
    ]

    # --- Create Table Plot ---
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
        cellText=data,
        colLabels=["Metric", "Value"],
        loc='center',
        cellLoc='center',
        colLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.title("Training Summary", fontweight="bold")
    plt.show()


# ---------------
# Main Function
# ---------------
def main():
    env = DotsAndBoxes(size=3)
    agent = QLearningAgent(name="Q-Agent")

    # Choose mode
    fictitious = False  # Set True for hybrid fictitious self-play, False for direct self-play

    # Train with population (hybrid option) or not (direct self-play)
    if fictitious:
        win_rates_self_play, win_rates_against_random, wins, score_margins = train_fsp(
            agent, env, episodes=10000, eval_interval=500
        )
    else:
        win_rates_self_play, win_rates_against_random, wins, score_margins = train_dsp(
            agent, env, episodes=10000, eval_interval=500
        )

    print("Final wins:", wins)
    visualize_results(win_rates_self_play, win_rates_against_random, wins, score_margins)

if __name__ == '__main__':
    main()