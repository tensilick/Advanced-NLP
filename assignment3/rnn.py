
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
        #  - node: your current node in the parse tree
        #  - correct: this is a running list of truth labels
        #  - guess: this is a running list of guess that our model makes
        #     (we will use both correct and guess to make our confusion matrix)
        ################

        #import pdb
        #pdb.set_trace()

        cost_right = 0
        cost_left  = 0
        tot_left   = 0
        tot_right  = 0

        if node.left != None and node.left.fprop == False:
            cost_left, tot_left = self.forwardProp(node.left,correct,guess)

        if node.right != None and node.right.fprop == False:
            cost_right, tot_right = self.forwardProp(node.right,correct,guess)

        cost  = cost_right + cost_left
        total = tot_left + tot_right

        if node.isLeaf:
            node.hAct1s = self.L[:,node.word]
        else:
            node.hAct1s = np.dot(self.W, np.hstack([node.left.hAct1s,node.right.hAct1s])) + self.b
            node.hAct1s[node.hAct1s < 0] = 0

        node.probs = np.dot(self.Ws,node.hAct1s) + self.bs
        node.probs -= np.max(node.probs)
        node.probs = np.exp(node.probs)
        node.probs = node.probs / np.sum(node.probs)

        cost -= np.log( node.probs[node.label] )

        correct.append(node.label)
        guess.append(np.argmax(node.probs))
        node.fprop = True

        return cost, total + 1


    def backProp(self,node,error=None):

        # Clear nodes
        node.fprop = False

        ################
        # TODO: Implement the recursive backProp function
        #  - you should update self.dWs, self.dbs, self.dW, self.db, and self.dL[node.word] accordingly