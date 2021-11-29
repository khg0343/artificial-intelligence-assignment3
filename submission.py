#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *



############################################################
# Problem 1: hinge loss
############################################################


def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        pretty, good, bad, plot, not, scenery
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    # raise NotImplementedError  # remove this line before writing code

    dict = collections.defaultdict(int)
    dict["pretty"] = 1
    dict["good"] = 0
    dict["bad"] = -1
    dict["plot"] = -1
    dict["not"] = -1
    dict["scenery"] = 0
    
    # print(dict)
    return dict
    # END_YOUR_ANSWER


############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction


def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 6 lines of code, but don't worry if you deviate from this)
    # raise NotImplementedError  # remove this line before writing code
    dict = collections.defaultdict(int)
    wordList = x.split()
    for word in wordList:
        if word not in dict:
            dict[word] = 1
        else :
            dict[word] += 1
    return dict

    # END_YOUR_ANSWER


############################################################
# Problem 2b: stochastic gradient descent


def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    """
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    """
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
    # raise NotImplementedError  # remove this line before writing code

    for x, y in trainExamples:
        for p in featureExtractor(x):
            weights[p] = 0
               
    for i in range(numIters):
        for x, y in trainExamples:
            phi = featureExtractor(x)
            dp = dotProduct(weights,phi)
            for p in phi:
                weights[p] += eta * phi[p] * (sigmoid(-dp) + (y-1)/2)
        
    # END_YOUR_ANSWER
    return weights

############################################################
# Problem 2c: bigram features


def extractBigramFeatures(x):
    """
    Extract unigram and bigcram features for a string x, where bigram feature is a tuple of two consecutive words. In addition, you should consider special words '<s>' and '</s>' which represent the start and the end of sentence respectively. You can exploit extractWordFeatures to extract unigram features.

    For example:
    >>> extractBigramFeatures("I am what I am")
    {('am', 'what'): 1, 'what': 1, ('I', 'am'): 2, 'I': 2, ('what', 'I'): 1, 'am': 2, ('<s>', 'I'): 1, ('am', '</s>'): 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
    # raise NotImplementedError  # remove this line before writing code

    phi = collections.defaultdict(int)
    wordList = x.split()
    for word in wordList:
        if word not in phi:
            phi[word] = 1
        else :
            phi[word] += 1
            
    for i in range(len(wordList)+1):
        if i == 0 :
            phi[('<s>', wordList[i])] = 1
        elif i == len(wordList) :
            phi[(wordList[len(wordList)-1], '</s>')] = 1
        elif (wordList[i-1], wordList[i]) not in phi:
            phi[(wordList[i-1], wordList[i])] = 1
        else :
            phi[(wordList[i-1], wordList[i])] += 1

    # END_YOUR_ANSWER
    return phi
