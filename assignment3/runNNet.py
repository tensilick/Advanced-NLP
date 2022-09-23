
import optparse
import cPickle as pickle

import sgd as optimizer
from rntn import RNTN
from rnn2deep import RNN2
from rnn import RNN
#from dcnn import DCNN
from rnn_changed import RNN3
import tree as tr
import time
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os
from sklearn.metrics import confusion_matrix

# This is the main training function of the codebase. You are intended to run this function via command line
# or by ./run.sh

# You should update run.sh accordingly before you run it!


# TODO:
# Create your plots here

def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--test",action="store_true",dest="test",default=False)

    # Optimizer
    parser.add_option("--minibatch",dest="minibatch",type="int",default=30)
    parser.add_option("--optimizer",dest="optimizer",type="string",
        default="adagrad")
    parser.add_option("--epochs",dest="epochs",type="int",default=50)
    parser.add_option("--step",dest="step",type="float",default=1e-2)


    parser.add_option("--middleDim",dest="middleDim",type="int",default=10)
    parser.add_option("--outputDim",dest="outputDim",type="int",default=5)
    parser.add_option("--wvecDim",dest="wvecDim",type="int",default=30)

    # By @tiagokv, just to ease the first assignment test
    parser.add_option("--wvecDimBatch",dest="wvecDimBatch",type="string",default="")
