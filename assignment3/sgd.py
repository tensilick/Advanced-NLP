import numpy as np
import random

class SGD:

    def __init__(self,model,alpha=1e-2,minibatch=30,
                 optimizer='sgd'):
        self.model = model
        print "initializing SGD"
        assert self.model is not None, "Must define a function to optimize"
        self.it = 0
        self.alpha