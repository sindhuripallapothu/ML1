import numpy as np
import pandas as pd
import random
eps = np.finfo(float).eps
import copy


df = pd.read_csv('training_set.csv')
dtest = pd.read_csv('test_set.csv')
dvalidation = pd.read_csv('validation_set.csv')

nodeCount = 0 # for node id
pruneFactor = 0.8

print("Please wait to complete!")

def find_variance(df):
    
    counts = df.groupby('Class').size()
    countone = counts[0]
    countzero = counts[1]
    total_count = len(df['Class'])
    variance = (countone/total_count)*(countzero/total_count)
    return variance



def find_variance_attribute(df,attribute):
  target_variables = df['Class'].unique()  #This gives all '1' and '0'
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


def find_winner_variance(df):
    Variance_att = []
    IGV = []
    for key in df.keys()[:-1]:
         Variance_att.append(find_variance_attribute(df,key))
         IGV.append(find_variance(df)-find_variance_attribute(df,key))
    
    return df.keys()[:-1][np.argmax(IGV)]

class Node():
    def __init__(self):
        self.left = None
        self.right = None
        self.attribute = None
        self.nodeType = None # L/R/I leaf/Root/Intermediate 
        self.value = None # attributes split's value 0 or 1
        self.positiveCount = None
        self.negativeCount = None
        self.label = None
        self.nodeId = None
    
    def setNodeValue(self, attribute, nodeType, value = None, positiveCount = None, negativeCount = None):
        self.attribute = attribute
        self.nodeType = nodeType
        self.value = value
        self.positiveCount = positiveCount
        self.negativeCount = negativeCount

class Tree():
    def __init__(self):
        self.root = Node()
        self.root.setNodeValue('$@$', 'R')
        
    def createDecisionTree(self, data, tree):
        global nodeCount
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
            bestAttribute = find_winner_variance(data)
            tree.left = Node()
            tree.right = Node()
            
            tree.left.nodeId = nodeCount
            nodeCount=nodeCount+1
            tree.right.nodeId = nodeCount
            nodeCount=nodeCount+1
            
            tree.left.setNodeValue(bestAttribute, 'I', 0, data[(data[bestAttribute]==0) & (df['Class']==1) ].shape[0], data[(data[bestAttribute]==0) & (df['Class']==0) ].shape[0])
            tree.right.setNodeValue(bestAttribute, 'I', 1, data[(data[bestAttribute]==1) & (df['Class']==1) ].shape[0], data[(data[bestAttribute]==1) & (df['Class']==0) ].shape[0])
            self.createDecisionTree( data[data[bestAttribute]==0].drop([bestAttribute], axis=1), tree.left)
            self.createDecisionTree( data[data[bestAttribute]==1].drop([bestAttribute], axis=1), tree.right)
            
    def printTreeLevels(self, node,level):
        if(node.left is None and node.right is not None):
            for i in range(0,level):    
                print("| ",end="")
            level = level + 1
            print("{} = {} (ID:{}) : {}".format(node.attribute, node.value,(node.nodeId if node.nodeId is not None else ""),(node.label if node.label is not None else "")))
            self.printTreeLevels(node.right,level)
        elif(node.right is None and node.left is not None):
            for i in range(0,level):    
                print("| ",end="")
            level = level + 1
#            print("{} = {} : {}".format(node.attribute, node.value,(node.label if node.label is not None else "")))
            print("{} = {} (ID:{}) : {}".format(node.attribute, node.value,(node.nodeId if node.nodeId is not None else ""),(node.label if node.label is not None else "")))
            self.printTreeLevels(node.left,level)
        elif(node.right is None and node.left is None):
            for i in range(0,level):    
                print("| ",end="")
            level = level + 1
            print("{} = {} (ID:{}) : {}".format(node.attribute, node.value,(node.nodeId if node.nodeId is not None else ""),(node.label if node.label is not None else "")))
        else:
            for i in range(0,level):    
                print("| ",end="")
            level = level + 1
            print("{} = {} (ID:{}) : {}".format(node.attribute, node.value,(node.nodeId if node.nodeId is not None else ""),(node.label if node.label is not None else "")))
            self.printTreeLevels(node.left,level)
            self.printTreeLevels(node.right,level)
    
    def printTree(self, node):
        self.printTreeLevels(node.left,0)
        self.printTreeLevels(node.right,0)
    
    def predictLabel(self, data, root):
        if root.label is not None:
            return root.label
        elif data[root.left.attribute][data.index.tolist()[0]] == 1:
            return self.predictLabel(data, root.right)
        else:
            return self.predictLabel(data, root.left)

    def countNodes(self,node):
        if(node.left is not None and node.right is not None):
            return 2 + self.countNodes(node.left) + self.countNodes(node.right)
        return 0

    
    def countLeaf(self,node):
        if(node.left is None and node.right is None):
            return 1
        return self.countLeaf(node.left) + self.countLeaf(node.right)

def searchNode(tree, x):
    tmp = None
    res = None
    if(tree.nodeType != "L"):
        if(tree.nodeId == x):
            return tree
        else:
            res = searchNode(tree.left,x)
            if (res is None):
                res = searchNode(tree.right,x)
            return res
    else:
        return tmp
    
    
def calculateAccuracy(data, tree):
    correctCount = 0
    for i in data.index:
        val = tree.predictLabel(data.iloc[i:i+1, :].drop(['Class'], axis=1),tree.root)
        if val == data['Class'][i]:
            correctCount = correctCount + 1
    return correctCount/data.shape[0]*100

def postPruning(pNum,newTree):
    
    for i in range(pNum):
        x = random.randint(2,pruneTree.countNodes(pruneTree.root)-1)
     
        tempNode = Node()
        tempNode = searchNode(newTree, x)

        if(tempNode is not None):
            tempNode.left = None
            tempNode.right = None
            tempNode.nodeType = "L"
            if(tempNode.negativeCount >= tempNode.positiveCount):
                tempNode.label = 0
            else:
                tempNode.label = 1
                
def postpruning1(L, K, dtree):
    dTreebest = copy.deepcopy(dtree)
    for i in range(L+1):
         pruneTree = copy.deepcopy(dtree)
         M = random.randint(1,K)
         for j in range(M+1):
             N = pruneTree.countLeaf(pruneTree.root)
             P =  random.randint(1,N)
             tempNode = searchNode(pruneTree.root, P)
             if(tempNode is not None):
                 tempNode.left = None
                 tempNode.right = None
                 tempNode.nodeType = "L"
                 if(tempNode.negativeCount >= tempNode.positiveCount):
                     tempNode.label = 0
                 else:
                     tempNode.label = 1
        
             maxAccuracypruneTree = calculateAccuracy(dvalidation, pruneTree)
             maxAccuracyTree = calculateAccuracy(dvalidation, dTreebest)
             if(maxAccuracypruneTree > maxAccuracyTree):
                 dTreebest = copy.deepcopy(pruneTree)
        
    return dTreebest


dtree = Tree()
dtree.createDecisionTree(df, dtree.root)
maxAccuracy = calculateAccuracy(dvalidation, dtree)

#bestTree = copy.deepcopy(dtree)
#countOfNodes = bestTree.countNodes(bestTree.root)
#c = 0
#
#while c < 10:    
#    c += 1
#    pruneNum = round(countOfNodes * pruneFactor)
#    
#    #print(pruneNum)
#    pruneTree = Tree()
#    pruneTree = copy.deepcopy(bestTree)
#    postPruning(pruneNum,pruneTree.root)
#    temp = calculateAccuracy(dvalidation, pruneTree)
#    if temp > maxAccuracy:
##         print("Accuracy Improved")
#        maxAccuracy = temp
#        bestTree = copy.deepcopy(pruneTree)
#        countOfNodes = bestTree.countNodes(bestTree.root)
        


print("-------------------------------------")
print("Pre-Pruned Tree")
print("-------------------------------------")
dtree.printTree(dtree.root)

print("")
print("-------------------------------------")
print("Pre-Pruned Accuracy")
print("-------------------------------------")
print("Number of training instances = " +str(df.shape[0]))
print("Number of training attributes = " +str(df.shape[1] -1))
print("Total number of nodes in the tree = "+ str(dtree.countNodes(dtree.root)))
print("Number of leaf nodes in the tree = " +str(dtree.countLeaf(dtree.root)))
print("Accuracy of the model on the training dataset = " + str(calculateAccuracy(df,dtree))+"%")
print("")
print("Number of validation instances = " +str(dvalidation.shape[0]))
print("Number of validation attributes = " +str(dvalidation.shape[1]-1))
print("Accuracy of the model on the validation dataset before pruning = "+ str(calculateAccuracy(dvalidation,dtree))+"%")
print("")
print("Number of testing instances = "+str(dtest.shape[0]))
print("Number of testing attributes = "+str(dtest.shape[1]-1))
print("Accuracy of the model on the testing dataset = "+ str(calculateAccuracy(dtest,dtree))+"%")

print("")
print("-------------------------------------")
print("Post-Pruned Tree")
print("-------------------------------------")
bestTree.printTree(bestTree.root)

print("")
print("-------------------------------------")
print("Post-Pruned Accuracy")
print("-------------------------------------")
print("Number of training instances = " + str(df.shape[0]))
print("Number of training attributes = " + str(df.shape[1] - 1))
print("Total number of nodes in the tree = " + str(bestTree.countNodes(bestTree.root)))
print("Number of leaf nodes in the tree = " + str(bestTree.countLeaf(bestTree.root)))
print("Accuracy of the model on the training dataset = " + str(calculateAccuracy(df, bestTree)) + "%")
print("")
print("Number of validation instances = " + str(dvalidation.shape[0]))
print("Number of validation attributes = " + str(dvalidation.shape[1] - 1))
print("Accuracy of the model on the validation dataset after pruning = " + str(calculateAccuracy(dvalidation, bestTree)) + "%")
print("")
print("Number of testing instances = " + str(dtest.shape[0]))
print("Number of testing attributes = " + str(dtest.shape[1] - 1))
print("Accuracy of the model on the testing dataset = " + str(calculateAccuracy(dtest, bestTree)) + "%")
print("")
if(maxAccuracy > calculateAccuracy(dvalidation, dtree)):
    print("Successfully Pruned with improvement in Accuracy on validation data set.")
else:
    print("Pruned but Accuracy didn't improved after 10 attempts so returning same tree.")