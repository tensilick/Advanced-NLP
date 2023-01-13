
import collections
UNK = 'UNK'
# This file contains the dataset in a useful way. We populate a list of Trees to train/test our Neural Nets such that each Tree contains any number of Node objects.

# The best way to get a feel for how these objects are used in the program is to drop pdb.set_trace() in a few places throughout the codebase
# to see how the trees are used.. look where loadtrees() is called etc..


class Node: # a node in the tree
    def __init__(self,label,word=None):
        self.label = label 
        self.word = word # NOT a word vector, but index into L.. i.e. wvec = L[:,node.word]
        self.parent = None # reference to parent
        self.left = None # reference to left child
        self.right = None # reference to right child
        self.isLeaf = False # true if I am a leaf (could have probably derived this from if I have a word)
        self.fprop = False # true if we have finished performing fowardprop on this node (note, there are many ways to implement the recursion.. some might not require this flag)
        self.hActs1 = None # h1 from the handout
        self.hActs2 = None # h2 from the handout (only used for RNN2)
        self.probs = None # yhat

class Tree:

    def __init__(self,treeString,openChar='(',closeChar=')'):
        tokens = []
        self.open = '('
        self.close = ')'
        for toks in treeString.strip().split():
            tokens += list(toks)
        self.root = self.parse(tokens)

    def parse(self, tokens, parent=None):
        assert tokens[0] == self.open, "Malformed tree"
        assert tokens[-1] == self.close, "Malformed tree"

        split = 2 # position after open and label
        countOpen = countClose = 0

        if tokens[split] == self.open: 
            countOpen += 1
            split += 1
        # Find where left child and right child split
        while countOpen != countClose:
            if tokens[split] == self.open:
                countOpen += 1
            if tokens[split] == self.close:
                countClose += 1
            split += 1

        # New node
        node = Node(int(tokens[1])) # zero index labels

        node.parent = parent 

        # leaf Node
        if countOpen == 0:
            node.word = ''.join(tokens[2:-1]).lower() # lower case?
            node.isLeaf = True
            return node

        node.left = self.parse(tokens[2:split],parent=node)
        node.right = self.parse(tokens[split:-1],parent=node)

        return node

        

def leftTraverse(root,nodeFn=None,args=None):
    """
    Recursive function traverses tree
    from left to right. 
    Calls nodeFn at each node
    """
    nodeFn(root,args)
    if root.left is not None:
        leftTraverse(root.left,nodeFn,args)
    if root.right is not None:
        leftTraverse(root.right,nodeFn,args)


def countWords(node,words):
    if node.isLeaf:
        words[node.word] += 1

def clearFprop(node,words):
    node.fprop = False

def mapWords(node,wordMap):
    if node.isLeaf:
        if node.word not in wordMap:
            node.word = wordMap[UNK]
        else:
            node.word = wordMap[node.word]
    

def loadWordMap():
    import cPickle as pickle
    
    with open('wordMap.bin','r') as fid:
        return pickle.load(fid)

def buildWordMap():
    """
    Builds map of all words in training set
    to integer values.
    """
