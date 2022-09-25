
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

    # for DCNN only
    parser.add_option("--ktop",dest="ktop",type="int",default=5)
    parser.add_option("--m1",dest="m1",type="int",default=10)
    parser.add_option("--m2",dest="m2",type="int",default=7)
    parser.add_option("--n1",dest="n1",type="int",default=6)
    parser.add_option("--n2",dest="n2",type="int",default=12)

    parser.add_option("--outFile",dest="outFile",type="string",
        default="models/test.bin")
    parser.add_option("--inFile",dest="inFile",type="string",
        default="models/test.bin")
    parser.add_option("--data",dest="data",type="string",default="train")

    parser.add_option("--model",dest="model",type="string",default="RNN")

    (opts,args)=parser.parse_args(args)


    # make this false if you dont care about your accuracies per epoch, makes things faster!
    evaluate_accuracy_while_training = True

    # Testing
    if opts.test:
        test(opts.inFile,opts.data,opts.model)
        return

    print "Loading data..."
    train_accuracies = []
    dev_accuracies = []
    # load training data
    trees = tr.loadTrees('train')
    opts.numWords = len(tr.loadWordMap())

    if (opts.model=='RNTN'):
        nn = RNTN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
    elif(opts.model=='RNN'):
        nn = RNN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
    elif(opts.model=='RNN2'):
        nn = RNN2(opts.wvecDim,opts.middleDim,opts.outputDim,opts.numWords,opts.minibatch)
    elif(opts.model=='RNN3'):
        nn = RNN3(opts.wvecDim,opts.middleDim,opts.outputDim,opts.numWords,opts.minibatch)
    elif(opts.model=='DCNN'):
        nn = DCNN(opts.wvecDim,opts.ktop,opts.m1,opts.m2, opts.n1, opts.n2,0, opts.outputDim,opts.numWords, 2, opts.minibatch,rho=1e-4)
        trees = cnn.tree2matrix(trees)
    else:
        raise '%s is not a valid neural network so far only RNTN, RNN, RNN2, RNN3, and DCNN'%opts.model
