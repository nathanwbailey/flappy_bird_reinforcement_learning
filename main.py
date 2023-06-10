import os
import sys
sys.path.append("./PyGame-Learning-Environment")
import pygame as pg
from ple import PLE 
from ple.games.flappybird import FlappyBird
from ple import PLE
import agent

game = FlappyBird(width=256, height=256)
p = PLE(game, display_screen=False)
p.init()
actions = p.getActionSet()
#List of possible actions is go up or do nothing
action_dict = {0: actions[1], 1: actions[0]}

#get the initial game state
state = p.getGameState()
len_state = len(state)
n_actions = len(action_dict)

#Create the agent and train it
agent = agent.Agent(BATCH_SIZE=32, MEMORY_SIZE=100000, GAMMA=0.99, input_dim=len_state, output_dim=n_actions, action_dim=n_actions, action_dict=action_dict, EPS_START=1.0, EPS_END=0.05, EPS_DECAY_VALUE=100000, TAU = 0.005, network_type='DuelingDDQN', lr = 1e-4)

agent.train(episodes=10000000, env=p)

