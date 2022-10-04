import numpy as np
import random

class SGD:

    def __init__(self,model,alpha=1e-2,minibatch=30,
                 optimizer='sgd'):
        self.model = model
       