import numpy as np
import random

class SGD:

    def __init__(self,model,alpha=1e-2,minibatch=30,
                 optimizer='sgd'):
        self.model = model
        print "initializing SGD"
        assert self.model is not None, "Must define a function to optimize"
        self.it = 0
        self.alpha = alpha # learning rate
        self.minibatch = minibatch # minibatch
        self.optimizer = optimizer
        if self.optimizer == 'sgd':
            print "Using sgd.."
        elif self.optimizer == 'adagrad':
            print "Using adagrad..."
            epsilon = 1e-8
            self.gradt = [epsilon + np.zeros(W.shape) for W in self.model.stack]
        else:
            raise ValueError("Invalid optimizer")

        self.costt = []
        self.expcost = []

    def run(self,trees):
        """
        Runs stochastic gradient descent w