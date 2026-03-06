# Dots and Boxes - MARL Analysis
## Reinforcement Learning Final Project

**By** Cullen McCaleb, Spring 2025

---
[Download the PDF Document](/docs/DotsAndBoxesReport.pdf)
---

## Overview
- dotbox.py is a python file that simulates a dots and boxes environment, and 
  uses two Q-learning algorithms to learn to play the game. One uses direct 
  self-play for learning and the other uses a variation of fictitious self-play
  that uses frozen-agents.
- The environment simulates a 3×3 grid of boxes (4×4 grid of dots) using two 2D 
  NumPy arrays: a 4×3 array for horizontal lines and a 3×4 array for vertical lines.
- The Q-table is implemented as a Python dictionary, where the keys are tuples 
  of the flattened game state (horizontal and vertical line arrays) and the 
  current player.
- The agent is trained using Q-learning with an epsilon-greedy action selection policy.
  The default value of epsilon is 0.1. The default for alpha and gamma in the update 
  rule is 0.5 and 0.9 respectively.
- The two modes of training can be toggles with a boolean in the main function.
- The program runs 10,000 episodes where each episode is one game. There is a +1 reward 
  for each completed box and a +10/-10 reward for winning or losing.
- Finally, the program visualizes the results using a win rate plot and a table 
  showing the results summary.

---

## Dependencies
- `numpy`  
- `matplotlib`
  
---

## Usage
1. Compile and run the code using:
```
   python3 dotbox.py
```
