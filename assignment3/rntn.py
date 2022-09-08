
import numpy as np
import collections
np.seterr(over='raise',under='raise')

class RNTN:

    def __init__(self,wvecDim,outputDim,numWords,mbSize=30,rho=1e-6):
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

        # Hidden activation weights
        self.V = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim,2*self.wvecDim)
        self.W = 0.01*np.random.randn(self.wvecDim,self.wvecDim*2)
        self.b = np.zeros((self.wvecDim))

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.wvecDim)
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.V, self.W, self.b, self.Ws, self.bs]

        # Gradients
        self.dV = np.empty((self.wvecDim,2*self.wvecDim,2*self.wvecDim))
        self.dW = np.empty(self.W.shape)
        self.db = np.empty((self.wvecDim))
        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))

    def costAndGrad(self,mbdata,test=False): 
        cost = 0.0
        correct = []
        guess = []
        total = 0.0

        return cost, []

    def forwardProp(self,node):
        cost = total = 0.0

        

        return cost,total + 1


    def backProp(self,node,error=None):

        # Clear nodes
        node.fprop = False

        
    def updateParams(self,scale,update,log=False):
        """
        Updates parameters as
        p := p - scale * update.
        If log is true, prints root mean square of parameter
        and update.
        """
        if log:
            for P,dP in zip(self.stack[1:],update[1:]):
                pRMS = np.sqrt(np.mean(P**2))
                dpRMS = np.sqrt(np.mean((scale*dP)**2))
                print "weight rms=%f -- update rms=%f"%(pRMS,dpRMS)

        self.stack[1:] = [P+scale*dP for P,dP in zip(self.stack[1:],update[1:])]

        # handle dictionary update sparsely
        dL = update[0]
        for j in dL.iterkeys():
            self.L[:,j] += scale*dL[j]

    def toFile(self,fid):
        import cPickle as pickle
        pickle.dump(self.stack,fid)

    def fromFile(self,fid):
        import cPickle as pickle
        self.stack = pickle.load(fid)

    def check_grad(self,data,epsilon=1e-6):

        cost, grad = self.costAndGrad(data)
        err1 = 0.0
        count = 0.0

        print "Checking dW... (might take a while)"
        for W,dW in zip(self.stack[1:],grad[1:]):
            W = W[...,None,None] # add dimension since bias is flat
            dW = dW[...,None,None] 
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    for k in xrange(W.shape[2]):
                        W[i,j,k] += epsilon
                        costP,_ = self.costAndGrad(data)
                        W[i,j,k] -= epsilon
                        numGrad = (costP - cost)/epsilon
                        err = np.abs(dW[i,j,k] - numGrad)
                        #print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dW[i,j,k],numGrad,err)
                        err1+=err
                        count+=1
        if 0.001 > err1/count:
            print "Grad Check Passed for dW"
        else: