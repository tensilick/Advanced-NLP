
import numpy as np
import collections

# This is a simple Recursive Neural Netowrk with one ReLU Layer and a softmax layer
# TODO: You must update the forward and backward propogation functions of this file.

# You can run this file via 'python rnn.py' to perform a gradient check!

# tip: insert pdb.set_trace() in places where you are unsure whats going on

class RNN:

    def __init__(self,wvecDim,outputDim,numWords,mbSize=30,rho=1e-4):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho

    def initParams(self):
        np.random.seed(12341)

        # Word vectors
        self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)

        # Hidden layer parameters
        self.W = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim)
        self.b = np.zeros((self.wvecDim))

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.wvecDim) # note this is " U " in the notes and the handout.. there is a reason for the change in notation
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.W, self.b, self.Ws, self.bs]

        # Gradients
        self.dW = np.empty(self.W.shape)
        self.db = np.empty((self.wvecDim))
        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))

    def costAndGrad(self,mbdata,test=False):
        """
        Each datum in the minibatch is a tree.
        Forward prop each tree.
        Backprop each tree.
        Returns
           cost
           Gradient w.r.t. W, Ws, b, bs
           Gradient w.r.t. L in sparse form.

        or if in test mode
        Returns
           cost, correctArray, guessArray, total

        """
        cost = 0.0
        correct = []
        guess = []
        total = 0.0

        self.L,self.W,self.b,self.Ws,self.bs = self.stack

        # Zero gradients
        self.dW[:] = 0
        self.db[:] = 0
        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dL = collections.defaultdict(self.defaultVec)

        # Forward prop each tree in minibatch
        for tree in mbdata:
            c,tot = self.forwardProp(tree.root,correct,guess)
            cost += c
            total += tot
        if test:
            return (1./len(mbdata))*cost,correct,guess,total

        # Back prop each tree in minibatch
        for tree in mbdata:
            self.backProp(tree.root)

        # scale cost and grad by mb size
        scale = (1./self.mbSize)
        for v in self.dL.itervalues():
            v *=scale

        # Add L2 Regularization
        cost += (self.rho/2)*np.sum(self.W**2)
        cost += (self.rho/2)*np.sum(self.Ws**2)

        return scale*cost,[self.dL,scale*(self.dW + self.rho*self.W),scale*self.db,
                           scale*(self.dWs+self.rho*self.Ws),scale*self.dbs]

    def forwardProp(self,node,correct=[], guess=[]):
        cost  =  total = 0.0 # cost should be a running number and total is the total examples we have seen used in accuracy reporting later
        ################
        # TODO: Implement the recursive forwardProp function
        #  - you should update node.probs, node.hActs1, node.fprop, and cost