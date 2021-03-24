#!/usr/bin/env python

import sys, os, re, json
import glob, shutil
import time


################
# Sanity check #
################
import numpy as np

fail = 0
counter = 0
testcases = []

from functools import wraps
import traceback

def prompt(msg):
    yn = raw_input(msg + " [y/n]: ")
    return yn.lower().startswith('y')

class testcase(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, func):
        global testcases

        @wraps(func)
        def wrapper():
            global counter
            global fail
            counter += 1
            print ">> Test %d (%s)" % (counter, self.name)
            try:
                func()
                print "[ok] Passed test %d (%s)" % (counter, self.name)
            except Exception as e:
                fail += 1
                print "[!!] Error on test %d (%s):" % (counter, self.name)
                traceback.print_exc()

        testcases.append(wrapper)
        return wrapper

##
# Part 0

##
# Part 1
@testcase("Part1: test random_weight_matrix")
def test_random_weight_matrix():
    from misc import random_weight_matrix
    A = random_weight_matrix(100,100)
    assert(A.shape == (100,100))

@testcase("Part1: initialize window model")
def ner_init():
    from nerwindow import WindowMLP
    np.random.seed(10)
    wv = np.random.randn(20,10)
    clf = WindowMLP(wv, windowsize=3,
                    dims = [None, 15, 3], rseed=10)

@testcase("Part1: test predict_proba()")
def ner_predict_proba():
    from nerwindow import WindowMLP
    np.random.seed(10)
    wv = np.random.randn(20,10)
    clf = WindowMLP(wv, windowsize=3,
                    dims = [None, 15, 3], rseed=10)
    p = clf.predict_proba([1,2,3])
    assert(len(p.flatten()) == 3)
    p = clf.predict_proba([[1,2,3], [2,3,4]])
    assert(np.ndim(p) == 2)
    assert(p.shape == (2,3))

@testcase("Part1: test compute_loss()")
def ner_predict_proba():
    from nerwindow import WindowMLP
    np.random.seed(10)
    wv = np.random.randn(20,10)
    clf = WindowMLP(wv, windowsize=3,
                    dims = [None, 15, 3], rseed=10)
    J = clf.compute_loss([1,2,3], 1)
    print "  dummy: J = %g" % J
    J = clf.compute_loss([[1,2,3], [2,3,4]], [0,1])
    print "  dummy: J = %g" % J

@testcase("Part1: NER prediction - dev set")
def ner_pred_dev():
    devpred = np.loadtxt("dev.predicted", dtype=int)
    assert(len(devpred) == 51362) # dev set length

@testcase("Part1: NER prediction - test set")
def ner_pred_test():
    testpred = np.loadtxt("test.predicted", dtype=int)
    assert(len(testpred) == 46435)

def setup_probing():
    num_to_word = dict(enumerate(
                       ["hello", "world", "i", "am", "a", "banana",
                        "there", "is", "no", "spoon"]))
    tagnames = ["O", "LOC", "MISC", "ORG", "PER"]
    num_to_tag = dict(enumerate(tagnames))

    from nerwindow import WindowMLP
    np.random.seed(10)
    wv = np.random.randn(10,50)
    clf = WindowMLP(wv, windowsize=3,
                    dims = [None, 100, 5], rseed=10)
    return clf, num_to_word, num_to_tag

@testcase("Part1.1 (a): verify output format")
def ner_probe_a():
    from part11probing import part_a, part_b, part_c
    clf, num_to_word, num_to_tag = setup_probing()
    s,w = part_a(clf, num_to_word, verbose=False)
    assert(len(s) == len(w))
    if type(s) == dict: # some students may have done this
        for k in s.keys(): assert(k in w)
        for k in w.keys(): assert(k in s)
        assert(len(s) >= 5)
    else: # list
        assert(len(s[0]) == len(w[0]))
        assert(len(s[0]) == 10)
        assert(type(w[0][0]) == str)


@testcase("Part1.1 (b): verify output format")
def ner_probe_b():
    from part11probing import part_a, part_b, part_c
    clf, num_to_word, num_to_tag = setup_probing()
    s,w = part_b(clf, num_to_word, num_to_tag, verbose=False)
    assert(len(s) == len(w))
    assert(len(s) == 5)
    assert(len(s[0]) == len(w[0]))
    assert(len(s[0]) == 10)
    assert(type(w[0][0]) == str)


@testcase("Part1.1 (c): verify output format")
def ner_probe_b():
    from part11probing impo