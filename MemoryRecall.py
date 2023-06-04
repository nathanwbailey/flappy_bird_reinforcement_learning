from collections import deque
import random

'''
MemoryRecall module, this is needed to cache data from the environment.

Then to randomly sample a batch when needed for training

'''

class MemoryRecall():
    def __init__(self, memory_size) -> None:
        self.memory_size=memory_size
        self.memory = deque(maxlen = self.memory_size)
    
    def cache(self, data):
        self.memory.append(data)

    def recall(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

