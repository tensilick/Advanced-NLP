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
    return yn.lo