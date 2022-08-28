
import numpy as np
import collections
import pdb


def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####

    e = np.sqrt( float(6) / (m+n) )
    A0 = np.random.uniform(-e,e,(m,n))

    #### END YOUR CODE ####
    assert(A0.shape == (m,n))