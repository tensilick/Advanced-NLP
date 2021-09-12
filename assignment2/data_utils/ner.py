##
# Utility functions for NER assignment
# Assigment 2, part 1 for CS224D
##

from utils import invert_dict
from numpy import *

def load_wv(vocabfile, wvfile):
    wv = loadtxt(wvfile, dtype=float)
    with op