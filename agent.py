'''
agent.py

the meat of the reinforcement learning program

create the 'agent' to interact with the environment

takes actions, and trains the model

'''

import torch
import model
import MemoryRecall
import numpy as np
import random
from itertools import count
import pygame as pg
import matplotlib.pyplot as plt
import torch.optim as optim

class Agent():
    def __init__(self, BATCH_SIZE, MEMORY_SIZE, GAMMA, input_dim, output_dim, action_dim, action_dict, EPS_START, EPS_END, EPS_DECAY_VALUE, lr, network_type='DDQN') -> None:
        #Set all the values up
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.MEMORY_SIZE=MEMORY_SIZE
        self.action_dim = action_dim
        self.action_dict = action_dict
        self.EPS_START=EPS_START
        self.EPS_END=EPS_END
        self.EPS_DECAY_VALUE=EPS_DECAY_VALUE
        self.eps = EPS_START
        #Select the GPU if we have one
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.episode_duration = []
        #Create the cache recall memory
        self.cache_recall = MemoryRecall.MemoryRecall(memory_size=MEMORY_SIZE)
        self.network_type=network_type
        #Create the dual Q networks - target and policy nets
