
from numpy import *
import itertools
import time
import sys

# Import NN utils
from nn.base import NNBase
from nn.math import softmax, sigmoid, make_onehot
from nn.math import MultinomialSampler, multinomial_sample
from misc import random_weight_matrix, sigmoid_grad, softmax


class RNNLM(NNBase):
    """
    Implements an RNN language model of the form:
    h(t) = sigmoid(H * h(t-1) + L[x(t)])
    y(t) = softmax(U * h(t))
    where y(t) predicts the next word in the sequence

    U = |V| * dim(h) as output vectors
    L = |V| * dim(h) as input vectors

    You should initialize each U[i,j] and L[i,j]
    as Gaussian noise with mean 0 and variance 0.1

    Arguments:
        L0 : initial input word vectors
        U0 : initial output word vectors
        alpha : default learning rate
        bptt : number of backprop timesteps
    """

    def __init__(self, L0, U0=None,
                 alpha=0.005, rseed=10, bptt=1):

        self.hdim = L0.shape[1] # word vector dimensions
        self.vdim = L0.shape[0] # vocab size
        param_dims = dict(H = (self.hdim, self.hdim),
                          U = L0.shape)
        # note that only L gets sparse updates
        param_dims_sparse = dict(L = L0.shape)
        NNBase.__init__(self, param_dims, param_dims_sparse)

        #### YOUR CODE HERE ####

        # Initialize word vectors
        # either copy the passed L0 and U0 (and initialize in your notebook)
        # or initialize with gaussian noise here

        if U0 is None:
            self.params.U = random.normal(0,0.1,*param_dims["U"])
        else:
            self.params.U = U0.copy()

        if L0 is None:
            self.sparams.L = random.normal(0,0.1,*param_dims["L"])
        else:
            self.sparams.L = L0.copy()

        # Initialize H matrix, as with W and U in part 1

        self.params.H = random_weight_matrix(*param_dims["H"])

        self.rseed = rseed
        self.bptt  = bptt
        self.alpha = alpha

        #### END YOUR CODE ####


    def _acc_grads(self, xs, ys):
        """
        Accumulate gradients, given a pair of training sequences:
        xs = [<indices>] # input words
        ys = [<indices>] # output words (to predict)

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.H += (your gradient dJ/dH)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # update row

        Per the handout, you should:
            - make predictions by running forward in time
                through the entire input sequence
            - for *each* output word in ys, compute the
                gradients with respect to the cross-entropy
                loss for that output word
            - run backpropagation-through-time for self.bptt
                timesteps, storing grads in self.grads (for H, U)
                and self.sgrads (for L)

        You'll want to store your predictions \hat{y}(t)
        and the hidden layer values h(t) as you run forward,
        so that you can access them during backpropagation.

        At time 0, you should initialize the hidden layer to
        be a vector of zeros.
        """

        # Expect xs as list of indices
        ns = len(xs)

        # make matrix here of corresponding h(t)
        # hs[-1] = initial hidden state (zeros)
        hs = zeros((ns+1, self.hdim))
        # predicted probas
        ps = zeros((ns, self.vdim))

        #### YOUR CODE HERE ####
        ##
        # Forward propagation
        for step in xrange(0,ns):
            # print "hs[step-1].shape %s" % (hs[step-1].shape,)
            # print "self.params.H.shape %s" % (self.params.H.shape,)
            # print "self.sparams.L.shape %s" % (self.sparams.L.shape,)
            # print "self.sparams.L[xs[step]].shape %s" % (self.sparams.L[xs[step]].shape,)
            a1 = self.params.H.dot(hs[step-1].T).T + self.sparams.L[xs[step]]
            a1 = expand_dims(a1,axis=0)
            h  = sigmoid( a1 )
            a2 = self.params.U.dot(h.T).T
            # print "h.flatten().shape %s" % (h.flatten().shape,)
            # print "a2.shape %s" % (a2.shape,)
            # print "self.params.U.shape %s" % (self.params.U.shape,)
            y_hat = softmax( a2 )

            # print "y_hat.shape %s" % (y_hat.shape,)

            hs[step] = h.flatten()
            ps[step] = y_hat

        ##
        # Backward propagation through time
        for step in xrange(ns-1,-1,-1):
            t = zeros( ps[step].shape )
            t[ys[step]] = 1
            delta_out = ps[step] - t
            self.grads.U += outer(hs[step],delta_out).T

            delta_hidden = delta_out.dot(self.params.U) * sigmoid_grad( hs[step] )

            for step_bp in xrange(step,step-self.bptt-1,-1):
                if step_bp < 0:
                    break
                self.grads.H  += outer(delta_hidden,hs[step_bp-1])
                self.sgrads.L[xs[step_bp]] = delta_hidden
                delta_hidden = delta_hidden.dot(self.params.H) * sigmoid_grad( hs[step_bp-1] )                

        #### END YOUR CODE ####



    def grad_check(self, x, y, outfd=sys.stderr, **kwargs):
        """
        Wrapper for gradient check on RNNs;
        ensures that backprop-through-time is run to completion,
        computing the full gradient for the loss as summed over
        the input sequence and predictions.

        Do not modify this function!
        """
        bptt_old = self.bptt
        self.bptt = len(y)
        print >> outfd, "NOTE: temporarily setting self.bptt = len(y) = %d to compute true gradient." % self.bptt
        NNBase.grad_check(self, x, y, outfd=outfd, **kwargs)
        self.bptt = bptt_old
        print >> outfd, "Reset self.bptt = %d" % self.bptt


    def compute_seq_loss(self, xs, ys):
        """
        Compute the total cross-entropy loss
        for an input sequence xs and output
        sequence (labels) ys.

        You should run the RNN forward,
        compute cross-entropy loss at each timestep,
        and return the sum of the point losses.
        """

        ns = len(xs)

        h_ant = zeros((1, self.hdim))

        J = 0
        #### YOUR CODE HERE ####
        for step in xrange(0,ns):
            # print "hs[step-1].shape %s" % (hs[step-1].shape,)
            # print "self.params.H.shape %s" % (self.params.H.shape,)
            # print "self.sparams.L.shape %s" % (self.sparams.L.shape,)
            # print "self.sparams.L[xs[step]].shape %s" % (self.sparams.L[xs[step]].shape,)
            a1 = self.params.H.dot(h_ant.T).T + self.sparams.L[xs[step]]
            h  = sigmoid( a1 )
            a2 = self.params.U.dot(h.T).T