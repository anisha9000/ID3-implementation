# course: TCSS555
# Homework 2
# date: 10/03/2017
# name: Martine De Cock
# description: Training and testing decision trees with discrete-values attributes

import sys
import math
import pandas as pd
#new imports
import numpy as np
import scipy.stats as stats
import math

class DecisionNode:

    # A DecisionNode contains an attribute and a dictionary of children. 
    # The attribute is either the attribute being split on, or the predicted label if the node has no children.
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}

    # Visualizes the tree
    def display(self, level = 0):
        if self.children == {}: # reached leaf level
            print(": ", self.attribute, end="")
        else:
            for value in self.children.keys():
                prefix = "\n" + " " * level * 4
                print(prefix, self.attribute, "=", value, end="")
                self.children[value].display(level + 1)
     
    # Predicts the target label for instance x
    def predicts(self, x):
        if self.children == {}: # reached leaf level
            return self.attribute
        value = x[self.attribute]
        subtree = self.children[value]
        return subtree.predicts(x)


# Illustration of functionality of DecisionNode class
def funTree():
    myLeftTree = DecisionNode('humidity')
    myLeftTree.children['normal'] = DecisionNode('no')
    myLeftTree.children['high'] = DecisionNode('yes')
    myTree = DecisionNode('wind')
    myTree.children['weak'] = myLeftTree
    myTree.children['strong'] = DecisionNode('no')
    return myTree

def calculateProbability(examples, target):
    print("Inside calculateProbability")
    targetLabels = examples[target].tolist()
    labelCount = len(targetLabels)
    if labelCount <= 1:
        return []

    probabilities = []
    counts = stats.itemfreq(targetLabels)
    
    for row in range(0,counts.shape[0]):
        probabilities.append([counts[row,0],int(counts[row,1])/labelCount])
    print(probabilities)
    return probabilities

def getEntropy(examples, target):
    print("Inside getEntropy")
    probabilityList = calculateProbability(examples, target)
    entropy = 0
    for row in probabilityList:
        logValue = math.log(row[1],2)
        product = row[1]*logValue
        entropy+=product
    entropy = (-entropy)
    print(entropy)
    return entropy


def getBestAttribute(examples, target, attributes):
    baseProbability = getEntropy(examples, target);
    
    print(baseProbability)


def id3(examples, target, attributes):
    print("id3 entry data:")
    print(type(examples))
    print(examples)
    print(target)
    print(attributes)
    print(type(attributes))
    print("")
    '''
    terminating condition goes here
    1. create root node. get the unique value and assign it to leaf.
    2. check if attributes is empty and create node with root.
    '''
    bestAttribute = getBestAttribute(examples, target, attributes)
    tree = funTree()
    return tree


####################   MAIN PROGRAM ######################

# Reading input data
train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])
target = sys.argv[3]
attributes = train.columns.tolist()
attributes.remove(target)

# Learning and visualizing the tree
tree = id3(train,target,attributes)
tree.display()

# Evaluating the tree on the test data
correct = 0
for i in range(0,len(test)):
    if str(tree.predicts(test.loc[i])) == str(test.loc[i,target]):
        correct += 1
print("\nThe accuracy is: ", correct/len(test))
