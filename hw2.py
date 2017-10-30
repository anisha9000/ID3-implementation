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

# calculate p[i] of unique target values for the example set i
def calculateProbability(examples, target):
    targetLabels = examples[target].tolist()
    labelCount = len(targetLabels)
    if labelCount <= 1:
        return []

    probabilities = []
    counts = stats.itemfreq(targetLabels)
    
    for row in range(0,counts.shape[0]):
        probabilities.append([counts[row,0],int(counts[row,1])/labelCount])
    
    return probabilities

# calculate sum of (-p[i]*log(p[i])) for example set i
def getEntropy(examples, target):
    # get list of p[i]
    probabilityList = calculateProbability(examples, target)
    entropy = 0
    for row in probabilityList:
        logValue = math.log(row[1],2)
        product = row[1]*logValue
        entropy+=product
    entropy = (-entropy)
    return entropy

# get the best attribute for example set i, which has the maximum information gain
def getBestAttribute(examples, target, attributes):
    # Entropy(S)
    baseEntropy = getEntropy(examples, target);
    originalLength = len(examples)
    informationGain = []
    
    #Divide instances based on attributes one by one to find best attribute
    for attribute in attributes:
        # group instances based on unique value
        groupedData = examples.groupby(attribute)
        totalEntropy = 0
        #entropy for each value of attribute
        for key,exampleSubset in groupedData:
            del exampleSubset[attribute]
            entropyOfSubset = getEntropy(exampleSubset,target)
            # sum of (|S[i]|/|S|) * Entropy(S[i])
            totalEntropy += (len(exampleSubset)/ originalLength)*entropyOfSubset
        #save the calculated entropy per attribute to determine best attribute
        informationGain.append([attribute,baseEntropy-totalEntropy])
        
    bestAttribute = max(informationGain, key=lambda x: x[1])
    return(bestAttribute[0])


def id3(examples, target, attributes):
    uniques = examples.apply(lambda x: x.nunique()).loc[target]
    if uniques==1:
        return DecisionNode(examples[target].iloc[0])
    if len(attributes) == 0:
        item_counts = examples[target].value_counts()
        max_item = item_counts.idxmax()
        return DecisionNode(max_item)
    
    bestAttribute = getBestAttribute(examples, target, attributes)
    attributes.remove(bestAttribute)
    groupedData = examples.groupby(bestAttribute)
    #create root node
    attributeRoot = DecisionNode(bestAttribute)
    
    for key,exampleSubset in groupedData:
        if len(exampleSubset) == 0:
            # TODO fix to get max frequency
            item_counts = exampleSubset[target].value_counts()
            max_item = item_counts.idxmax()
            attributeRoot.children[key] = DecisionNode(max_item)
        else:
            attributeRoot.children[key] = id3(exampleSubset.drop([bestAttribute],axis=1), target, attributes)

    return attributeRoot


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
