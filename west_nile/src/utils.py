__author__ = 'alex'

import numpy as np

class AdjustVariable(object):
    def __init__(self, variable, target, half_life=20):
        self.variable = variable
        self.target = target
        self.half_life = half_life

    def __call__(self, nn, train_history):
        delta = self.variable.get_value() - self.target
        delta /= 2 ** (1.0 / self.half_life)
        self.variable.set_value(np.float32(self.target + delta))
