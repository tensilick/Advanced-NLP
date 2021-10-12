
import sys
from numpy import *
import numpy as np
import itertools
import time

# math helpers
from math import *

class PackedVector(object):
    _name_to_idx = {}
    _views = []

    def __init__(self, *shapes, **shapemap):
        # Prepend named shapes
        names = range(len(shapes))
        if len(shapemap) > 0:
            nn, ss = zip(*shapemap.items())
            names = names + list(nn)
            shapes = shapes + ss
        self._name_to_idx = {n:i for i,n in enumerate(names)}

        # Generate endpoints
        self._dims = shapes
        self._lens = map(prod, self._dims)
        self._ends = concatenate([[0], cumsum(self._lens)])
        self._vec = zeros(self._ends[-1]) # allocate storage
