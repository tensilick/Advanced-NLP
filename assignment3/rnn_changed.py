
import numpy as np
import collections
import pdb


def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####

    e = np.sqrt( float(6) / (m+n) )
    A0 = np.random.uniform(-e,e,(m,n))

    #### END YOUR CODE ####
    assert(A0.shape == (m,n))
    return A0

class RNN3:

    def __init__(self,wvecDim, middleDim, outputDim,numWords,mbSize=30,rho=1e-4):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.middleDim = middleDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho

    def initParams(self):
        np.random.seed(12341)

        # Word vectors
        self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)

        # Hidden activation weights for layer 1
        self.W1 = 0.01*np.random.randn(self.wvecDim,2*self.middleDim)
        #self.W1 = random_weight_matrix(self.wvecDim,2*self.wvecDim)
        self.b1 = np.zeros((self.wvecDim))

        # Hidden activation weights for layer 2
        self.W2 = 0.01*np.random.randn(self.middleDim,self.wvecDim)
        self.b2 = np.zeros((self.middleDim))

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.middleDim) # note this is " U " in the notes and the handout.. there is a reason for the change in notation
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.W1, self.b1, self.W2, self.b2, self.Ws, self.bs]

        # Gradients
        self.dW1 = np.empty(self.W1.shape)
        self.db1 = np.empty((self.wvecDim))

        self.dW2 = np.empty(self.W2.shape)
        self.db2 = np.empty((self.middleDim))

        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))


    def costAndGrad(self,mbdata,test=False):
        """
        Each datum in the minibatch is a tree.
        Forward prop each tree.
        Backprop each tree.
        Returns
           cost
           Gradient w.r.t. W1, W2, Ws, b1, b2, bs
           Gradient w.r.t. L in sparse form.

        or if in test mode
        Returns
           cost, correctArray, guessArray, total
        """
        cost = 0.0
        correct = []
        guess = []
        total = 0.0

        self.L, self.W1, self.b1, self.W2, self.b2, self.Ws, self.bs = self.stack
        # Zero gradients
        self.dW1[:] = 0
        self.db1[:] = 0

        self.dW2[:] = 0
        self.db2[:] = 0

        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dL = collections.defaultdict(self.defaultVec)

        # Forward prop each tree in minibatch
        for tree in mbdata:
            c,tot = self.forwardProp(tree.root,correct,guess)
            cost += c
            total += tot

        if test:
            return (1./len(mbdata))*cost,correct, guess, total

        # Back prop each tree in minibatch
        for tree in mbdata:
            self.backProp(tree.root)

        # scale cost and grad by mb size
        scale = (1./self.mbSize)
        for v in self.dL.itervalues():
            v *=scale

        # Add L2 Regularization
        cost += (self.rho/2)*np.sum(self.W1**2)
        cost += (self.rho/2)*np.sum(self.W2**2)
        cost += (self.rho/2)*np.sum(self.Ws**2)

        return scale*cost,[self.dL,scale*(self.dW1 + self.rho*self.W1),scale*self.db1,
                                   scale*(self.dW2 + self.rho*self.W2),scale*self.db2,
                                   scale*(self.dWs+self.rho*self.Ws),scale*self.dbs]


    def forwardProp(self,node, correct=[], guess=[]):
        cost  =  total = 0.0

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
            node.hActs1 = self.L[:,node.word]
            total = 0.0
        else:
            #ReLU
            node.hActs1 = np.dot(self.W1, np.hstack([node.left.hActs2,node.right.hActs2])) + self.b1 # Here is one of the main changes the prop is from h2 to h1 from the top layer
            node.hActs1[node.hActs1 < 0] = 0

        #ReLU
        node.hActs2 = np.dot(self.W2,node.hActs1) + self.b2
        node.hActs2[node.hActs2 < 0] = 0

        #Softmax layer
        node.probs = np.dot(self.Ws,node.hActs2) + self.bs
        node.probs -= np.max(node.probs)
        node.probs = np.exp(node.probs)
        node.probs = node.probs / np.sum(node.probs)

        #cost
        cost -= np.log( node.probs[node.label] )

        correct.append(node.label)
        guess.append(np.argmax(node.probs))
        node.fprop = True

        return cost, total + 1

    def backProp(self,node,error=None):

        # Clear nodes
        node.fprop = False

        deltas = node.probs
        deltas[node.label] -= 1.0
        self.dWs += np.outer(deltas,node.hActs2)
        self.dbs += deltas

        deltas = np.dot(self.Ws.T,deltas)

        if error is not None:
            deltas += error

        deltas *= (node.hActs2 != 0)

        self.dW2 += np.outer(deltas,node.hActs1)
        self.db2 += deltas

        deltas = np.dot(self.W2.T,deltas)

        deltas *= (node.hActs1 != 0)

        if node.isLeaf:
            self.dL[node.word] += deltas
            return
        else:
            self.dW1 += np.outer(deltas,np.hstack([node.left.hActs2,node.right.hActs2]))
            self.db1 += deltas
            deltas  = np.dot(self.W1.T,deltas)
            self.backProp(node.left, deltas[:self.middleDim])
            self.backProp(node.right,deltas[self.middleDim:])
