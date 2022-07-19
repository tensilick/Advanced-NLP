
#!/usr/bin/env python

import sys, os
from numpy import *

def print_scores(scores, words):
    for i in range(len(scores)):
        print "[%d]: (%.03f) %s" % (i, scores[i], words[i])


def part_a(clf, num_to_word, verbose=True):
    """
    Code for 1.1 part (a):
    Hidden Layer, Center Word

    clf: instance of WindowMLP,
            trained on data
    num_to_word: dict {int:string}

    You need to create:
    - topscores : list of lists of 10 scores (float)
    - topwords  : list of lists of 10 words (string)
    You should generate these lists for each neuron
    (so for hdim = 100, you'll have lists of 100 lists of 10)
    then fill in neurons = [<your chosen neurons>] to print
    """
    #### YOUR CODE HERE ####





    neurons = [1,3,4,6,8] # change this to your chosen neurons

    #### END YOUR CODE ####
    # topscores[i]: list of floats
    # topwords[i]: list of words
    if verbose == True:
        for i in neurons:
            print "Neuron %d" % i
            print_scores(topscores[i], topwords[i])

    return topscores, topwords


def part_b(clf, num_to_word, num_to_tag, verbose=True):
    """
    Code for 1.1 part (b):
    Model Output, Center Word

    clf: instance of WindowMLP,
            trained on data
    num_to_word: dict {int:string}

    You need to create:
    - topscores : list of 5 lists of 10 probability scores (float)
    - topwords  : list of 5 lists of 10 words (string)
    where indices 0,1,2,3,4 correspond to num_to_tag, i.e.
    tagnames = ["O", "LOC", "MISC", "ORG", "PER"]
    """
    #### YOUR CODE HERE ####






    #### END YOUR CODE ####
    # topscores[i]: list of floats
    # topwords[i]: list of words
    if verbose == True:
        for i in range(1,5):
            print "Output neuron %d: %s" % (i, num_to_tag[i])
            print_scores(topscores[i], topwords[i])
            print ""

    return topscores, topwords


def part_c(clf, num_to_word, num_to_tag, verbose=True):
    """
    Code for 1.1 part (c):
    Model Output, Preceding Word

    clf: instance of WindowMLP,
            trained on data
    num_to_word: dict {int:string}

    You need to create:
    - topscores : list of 5 lists of 10 probability scores (float)
    - topwords  : list of 5 lists of 10 words (string)
    where indices 0,1,2,3,4 correspond to num_to_tag, i.e.
    tagnames = ["O", "LOC", "MISC", "ORG", "PER"]
    """
    #### YOUR CODE HERE ####






    #### END YOUR CODE ####
    # topscores[i]: list of floats
    # topwords[i]: list of words
    if verbose == True:
        for i in range(1,5):
            print "Output neuron %d: %s" % (i, num_to_tag[i])
            print_scores(topscores[i], topwords[i])
            print ""