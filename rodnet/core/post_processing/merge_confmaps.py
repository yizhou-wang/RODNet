import numpy as np


class ConfmapStack:
    def __init__(self, confmap_shape):
        self.confmap = np.zeros(confmap_shape)
        self.count = 0
        self.next = None
        self.ready = False

    def append(self, confmap):
        self.confmap = (self.confmap * self.count + confmap) / (self.count + 1)
        self.count += 1

    def setNext(self, _genconfmap):
        self.next = _genconfmap
