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
print(actions)
#List of possible actions is go up or do nothing
action_dict = {0: actions[0], 1: actions[1]}

#get the initial ganme state
state = p.getGameState()
len_state = len(state)
n_actions = len(action_dict)

