
from numpy import *

def sigmoid(x):
    return 1.0/(1.0 + exp(-x))

def softmax(x):
    xt = exp(x - max(x))
    return xt / sum(xt)

def make_onehot(i, n):
    y = zeros(n)
    y[i] = 1
    return y


class MultinomialSampler(object):
    """