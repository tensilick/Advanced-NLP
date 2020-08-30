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
                print "[!!] Error on test %d (%s):" % (counter, self