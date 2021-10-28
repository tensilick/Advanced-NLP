
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

        # Generate view objects
        self._views = []
        for i, dim in enumerate(self._dims):
            start = self._ends[i]
            end = self._ends[i+1]
            self._views.append(self._vec[start:end].reshape(dim))

        # Add view for full params
        self._name_to_idx['full'] = len(self._views)
        self._views.append(self._vec[:]) # full

        # Generate attributes for direct access
        for n,i in self._name_to_idx.iteritems():
            object.__setattr__(self, str(n), self._views[i])

    # Overload setattr to write to views
    def __setattr__(self, name, value):
        if name in self._name_to_idx:
            v = self._views[self._name_to_idx[name]]
            v[:] = value # in-place update
        else: # default behavior
            object.__setattr__(self, name, value)

    ##
    # Dictionary-style access
    def __getitem__(self, key):
        key = self._name_to_idx[key]
        return self._views[key]

    def __setitem__(self, key, value):
        key = self._name_to_idx[key]
        self._views[key][:] = value # in-place update

    def names(self):
        return [k for k in self._name_to_idx.keys() if not k == 'full']

    def reset(self):
        self.full.fill(0)

    def __repr__(self):
        listings = ["%s = \n%s" % (n, repr(self._views[i]))
                    for (n,i) in sorted(self._name_to_idx.items())
                    if n != 'full']
        return "PackedVector(\n" + "\n".join(listings) + "\n)"


class SparseDelta(object):
    """
    Wrapper class for sparse updates;
    stores a list of (idx, value) tuples,
    while supporting dict-like syntax.

    idx can be any python object, so complex
    slicing/selection like [:,range(5)] will be
    handled properly, as this is just forwarded
    to the NumPy array later.
    """