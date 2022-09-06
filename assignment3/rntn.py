
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