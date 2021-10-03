
import sys, os, re, json
import itertools
from collections import Counter
import time
from numpy import *

import pandas as pd


def invert_dict(d):
    return {v:k for k,v in d.iteritems()}

def flatten1(lst):
    return list(itertools.chain.from_iterable(lst))

def load_wv_pandas(fname):
    return pd.read_hdf(fname, 'data')

def extract_wv(df):
    num_to_word = dict(enumerate(df.index))
    word_to_num = invert_dict(num_to_word)
    wv = df.as_matrix()
    return wv, word_to_num, num_to_word

def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "") # remove thousands separator
    return word

def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word) # try to canonicalize numbers
    if (wordset == None) or (word in wordset): return word
    else: return "UUUNKKK" # unknown token


##
# Utility functions used to create dataset
##
def augment_wv(df, extra=["UUUNKKK"]):
    for e in extra:
        df.loc[e] = zeros(len(df.columns))

def prune_wv(df, vocab, extra=["UUUNKKK"]):
    """Prune word vectors to vocabulary."""
    items = set(vocab).union(set(extra))
    return df.filter(items=items, axis='index')

def load_wv_raw(fname):
    return pd.read_table(fname, sep="\s+",
                         header=None,
                         index_col=0,
                         quoting=3)

def load_dataset(fname):
    docs = []
    with open(fname) as fd:
        cur = []
        for line in fd:
            # new sentence on -DOCSTART- or blank line
            if re.match(r"-DOCSTART-.+", line) or (len(line.strip()) == 0):
                if len(cur) > 0:
                    docs.append(cur)
                cur = []
            else: # read in tokens
                cur.append(line.strip().split("\t",1))
        # flush running buffer
        docs.append(cur)
    return docs

def extract_tag_set(docs):
    tags = set(flatten1([[t[1].split("|")[0] for t in d] for d in docs]))
    return tags

def extract_word_set(docs):
    words = set(flatten1([[t[0] for t in d] for d in docs]))
    return words

def pad_sequence(seq, left=1, right=1):
    return left*[("<s>", "")] + seq + right*[("</s>", "")]

##
# For window models
def seq_to_windows(words, tags, word_to_num, tag_to_num, left=1, right=1):
    ns = len(words)
    X = []
    y = []
    for i in range(ns):
        if words[i] == "<s>" or words[i] == "</s>":
            continue # skip sentence delimiters
        tagn = tag_to_num[tags[i]]
        idxs = [word_to_num[words[ii]]
                for ii in range(i - left, i + right + 1)]
        X.append(idxs)
        y.append(tagn)
    return array(X), array(y)

def docs_to_windows(docs, word_to_num, tag_to_num, wsize=3):
    pad = (wsize - 1)/2
    docs = flatten1([pad_sequence(seq, left=pad, right=pad) for seq in docs])

    words, tags = zip(*docs)
    words = [canonicalize_word(w, word_to_num) for w in words]
    tags = [t.split("|")[0] for t in tags]
    return seq_to_windows(words, tags, word_to_num, tag_to_num, pad, pad)

def window_to_vec(window, L):
    """Concatenate word vectors for a given window."""
    return concatenate([L[i] for i in window])

##
# For fixed-window LM:
# each row of X is a list of word indices
# each entry of y is the word index to predict
def seq_to_lm_windows(words, word_to_num, ngram=2):
    ns = len(words)
    X = []
    y = []
    for i in range(ns):
        if words[i] == "<s>":
            continue # skip sentence begin, but do predict end
        idxs = [word_to_num[words[ii]]
                for ii in range(i - ngram + 1, i + 1)]
        X.append(idxs[:-1])
        y.append(idxs[-1])
    return array(X), array(y)