
from nn.base import NNBase
from nn.math import softmax, make_onehot
from misc import random_weight_matrix
from numpy import *

class SoftmaxRegression(NNBase):
    """
    Dummy example, to show how to implement a network.
    This implements softmax regression, trained by SGD.
    """
