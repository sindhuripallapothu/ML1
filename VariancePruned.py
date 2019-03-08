# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 13:49:57 2019

@author: Sindhu
"""
import numpy as np
import pandas as pd
import random
import copy
import sys
import warnings

eps = np.finfo(float).eps


warnings.simplefilter("ignore", UserWarning)

if(len(sys.argv) != 7):
    sys.exit("Invalid Number of Arguments. - <L> <K> <training-set> <test-set> <validation-set>  <to-print>")
else:
    L1 = sys.argv[1]
    K1 = sys.argv[2]
    training_set = sys.argv[3]
    test_set = sys.argv[4]
    validation_set = sys.argv[5]
    toprint = sys.argv[6]

df = pd.read_csv(training_set)
dtest = pd.read_csv(test_set)
dvalidation = pd.read_csv(validation_set)
L = int(L1)
K= int(K1)


#df = pd.read_csv('training_set.csv')
#dtest = pd.read_csv('test_set.csv')
#dvalidation = pd.read_csv('validation_set.csv')
#L = 5
#K = 10
#toprint = 'Y'

node_number = 0 


print("....Variance Impurity Heuristic....."+ "\n"+ "\n")

def find_variance_target(df):
    
    counts = df.groupby('Class').size()
    countone = counts[0]
    countzero = counts[1]
    total_count = len(df['Class'])
    variance = (countone/total_count)*(countzero/total_count)
    return variance



def find_variance_attribute(df,attribute):
  target_variables = df['Class'].unique()  
  variables = df[attribute].unique()   
  variance2 = 0
  for variable in variables:
      variance = 1
      for target_variable in target_variables:
          num = len(df[attribute][df[attribute]==variable][df['Class'] ==target_variable])
          den = len(df[attribute][df[attribute]==variable])
          fraction = num/(den+eps)
          variance *= fraction
      fraction2 = den/len(df)
      variance2 += fraction2*variance
  
  return variance2


def find_best_variance_attribute(df):
    Variance_att = []
    IGV = []
    for key in df.keys()[:-1]:
         Variance_att.append(find_variance_attribute(df,key))
         IGV.append(find_variance_target(df)-find_variance_attribute(df,key))
    
    return df.keys()[:-1][np.argmax(IGV)]

class Node():
    def __init__(self):
        self.left = None
        self.right = None
        self.attribute = None
        self.nodeType = None # L/R/I leaf/Root/Intermediate 
        self.value = None 
        self.positiveCount = None
        self.negativeCount = None
        self.label = None
        self.nodeId = None
    
    def setValues(self, attribute, nodeType, value = None, positiveCount = None, negativeCount = None):
        self.attribute = attribute
        self.nodeType = nodeType
        self.value = value
        self.positiveCount = positiveCount
        self.negativeCount = negativeCount

class Tree():
    def __init__(self):
        self.root = Node()
        self.root.setValues('***', 'R')
        
    def createDecisionTree(self, data, tree):
        global node_number
        total = data.shape[0]
        ones = data['Class'].sum()
        zeros = total - ones        
        if data.shape[1] == 1 or total == ones or total == zeros:
            tree.nodeType = 'L'
            if zeros >= ones:
                tree.label = 0
            else:
                tree.label = 1
            return        
        else:        
            bestAttribute = find_best_variance_attribute(data)
            tree.left = Node()
            tree.right = Node()
            
            tree.left.nodeId = node_number
            node_number=node_number+1
            tree.right.nodeId = node_number
            node_number=node_number+1
            
            tree.left.setValues(bestAttribute, 'I', 0, data[(data[bestAttribute]==0) & (df['Class']==1) ].shape[0], data[(data[bestAttribute]==0) & (df['Class']==0) ].shape[0])
            tree.right.setValues(bestAttribute, 'I', 1, data[(data[bestAttribute]==1) & (df['Class']==1) ].shape[0], data[(data[bestAttribute]==1) & (df['Class']==0) ].shape[0])
            self.createDecisionTree( data[data[bestAttribute]==0].drop([bestAttribute], axis=1), tree.left)
            self.createDecisionTree( data[data[bestAttribute]==1].drop([bestAttribute], axis=1), tree.right)
            
    def printTreestandard(self, node,level):
        if(node.left is None and node.right is not None):
            for i in range(0,level):    
                print("| ",end="")
            level = level + 1
            print("{} = {}  : {}".format(node.attribute, node.value,(node.label if node.label is not None else "")))
            self.printTreestandard(node.right,level)
        elif(node.right is None and node.left is not None):
            for i in range(0,level):    
                print("| ",end="")
            level = level + 1
            print("{} = {}  : {}".format(node.attribute, node.value,(node.label if node.label is not None else "")))
            self.printTreestandard(node.left,level)
        elif(node.right is None and node.left is None):
            for i in range(0,level):    
                print("| ",end="")
            level = level + 1
            print("{} = {}  : {}".format(node.attribute, node.value,(node.label if node.label is not None else "")))
        else:
            for i in range(0,level):    
                print("| ",end="")
            level = level + 1
            print("{} = {}  : {}".format(node.attribute, node.value,(node.label if node.label is not None else "")))
            self.printTreestandard(node.left,level)
            self.printTreestandard(node.right,level)
    
    def printTree(self, node):
        self.printTreestandard(node.left,0)
        self.printTreestandard(node.right,0)
    
    def predict(self, data, root):
        if root.label is not None:
            return root.label
        elif data[root.left.attribute][data.index.tolist()[0]] == 1:
            return self.predict(data, root.right)
        else:
            return self.predict(data, root.left)

    def count_nodes(self,node):
        if(node.left is not None and node.right is not None):
            return 2 + self.count_nodes(node.left) + self.count_nodes(node.right)
        return 0

    
    def count_leaves(self,node):
        if(node.left is None and node.right is None):
            return 1
        return self.count_leaves(node.left) + self.count_leaves(node.right)

def search_node(tree, x):
    temp = None
    res = None
    if(tree.nodeType != "L"):
        if(tree.nodeId == x):
            return tree
        else:
            res = search_node(tree.left,x)
            if (res is None):
                res = search_node(tree.right,x)
            return res
    else:
        return temp
    
    
def calculateAccuracy(data, tree):
    correctCount = 0
    for i in data.index:
        val = tree.predict(data.iloc[i:i+1, :].drop(['Class'], axis=1),tree.root)
        if val == data['Class'][i]:
            correctCount = correctCount + 1
    return correctCount/data.shape[0]*100

                
def PostPruning(L, K, dtree):
    dTreebest = copy.deepcopy(dtree)
    for i in range(L+1):
         pruneTree = copy.deepcopy(dtree)
         M = random.randint(1,K)
         for j in range(M+1):
             N = pruneTree.count_leaves(pruneTree.root)
             P =  random.randint(1,N)
             tempNode = search_node(pruneTree.root, P)
             if(tempNode is not None):
                 tempNode.left = None
                 tempNode.right = None
                 tempNode.nodeType = "L"
                 if(tempNode.negativeCount >= tempNode.positiveCount):
                     tempNode.label = 0
                 else:
                     tempNode.label = 1
                   
                 dTreebest = copy.deepcopy(pruneTree)
        
    return dTreebest

####################################################################################
#Unpruned Tree
dtree = Tree()
dtree.createDecisionTree(df, dtree.root)
maxAccuracy1 = calculateAccuracy(dtest, dtree)


print("UnPruned Tree"+ "\n")
if (toprint in ('YES' , 'yes' , 'Y' , 'y')):
    dtree.printTree(dtree.root)
    
else: 
    print("OK, Unpruned Tree won't be printed.")


print("Unpruned Tree Accuracy")
print("Total number of nodes in the tree = "+ str(dtree.count_nodes(dtree.root)))
print("Accuracy of the model on the testing dataset = "+ str(calculateAccuracy(dtest,dtree))+"%")
print("Accuracy of the model on the validation dataset before pruning = "+ str(calculateAccuracy(dvalidation,dtree))+"%" + "\n"+ "\n")

####################################################################################
#Post Pruned tree
bestTree = PostPruning(L,K,dtree)
maxAccuracy2 = calculateAccuracy(dtest, bestTree)

print("Post-Pruned Tree"+ "\n")
if (toprint in ('YES' , 'yes' , 'Y' , 'y')):
    bestTree.printTree(bestTree.root)
    
else: 
    print("OK, Post Pruned Tree won't be printed.")

print("Post-Pruned Tree Accuracy")
print("Total number of nodes in the tree = " + str(bestTree.count_nodes(bestTree.root)))
print("Accuracy of the model on the testing dataset = " + str(calculateAccuracy(dtest, bestTree)) + "%")
print("Accuracy of the model on the validation dataset after pruning = " + str(calculateAccuracy(dvalidation, bestTree)) + "%"+ "\n"+ "\n")

####################################################################################
#comparing the accuracies on the validation set between pruned and unpruned tree.
if(maxAccuracy2 > maxAccuracy1):
    print("Accuracy has improved after pruning as compared to the unpruned tree "+ "\n"+ "\n")
else:
    print("Accuracy didn't improve."+ "\n"+ "\n")
####################################################################################
