The decision tree were created based on 2 Heuristics - 
-- Information Gain Heuristic
-- Variance Impurity Heuristic

I have divided the assignment into 2 .py file -
-- EntropyPruned.py
-- VariancePruned.py

Both the programs can be run of command prompt using 6 aruments - 

<L> <K> <training_set.csv> <testing_set.csv> <validation_set.csv> <toprint>

-- L, K - inputs to the post pruning algorithm.
-- training_set.csv, testing_set.csv, validation_set.csv - they are 3 the CSV files needed to
   train, test and validate the decision tree which is based on the 2 heuristics.
-- toprint - If the value is "YES" or "yes" or "Y" or "y", the decision tree will be printed 
   on the output console, any other value will not print the decsion tree.


Following is the output of the programs
-- Decision tree (in the standard output format if the toprint = "YES" or "yes" or "Y" or "y")
-- Total number of nodes in the decision tree
-- Accuracy on the testing data
-- Accuracy of the validation data.


 