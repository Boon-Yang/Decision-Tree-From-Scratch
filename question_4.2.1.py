import numpy as np
import reader
import metrics
from classification import DecisionTreeClassifier    

# Load data

x, y = reader.read_from_csv("data/train_full.txt")
x_val, y_val = reader.read_from_csv("data/validation.txt")
x_test, y_test = reader.read_from_csv("data/test.txt")

# 1. Train grid search on prepruning

tree_preprune = DecisionTreeClassifier()
options = {"max_tree_depth": [13, 15, 17], "min_sample_size": [2, 3, 4]}
print(metrics.grid_search(tree_preprune, x, y, x_val, y_val, options))
acc1 = metrics.accuracy(tree_preprune.predict(x_test), y_test)

# 2. Train existing classifier on postpruning (call post prune on 1)

tree_preprune.prune(x_val, y_val)
acc2 = metrics.accuracy(tree_preprune.predict(x_test), y_test)


# 3. Train new classifier on postpruning

tree_postprune = DecisionTreeClassifier()
tree_postprune.fit(x, y)
tree_postprune.prune(x_val, y_val)
acc3 = metrics.accuracy(tree_postprune.predict(x_test), y_test)

print("")
print("Results:")
print("preprune = ", acc2)
print("post prune existing = ", acc2)
print("post prune new = ", acc3)


# Note: Nothing was pruned all post pruning proccesss

"""
[arbitrary, arbitrary]

[5 5]
Best accuracy is 0.7
[ 5 20]
Best accuracy is 0.7117948717948718
[  5 100]
Best accuracy is 0.7158974358974359
[20  5]
Best accuracy is 0.8917948717948718
[20 20]
Best accuracy is 0.8533333333333334
[ 20 100]
Best accuracy is 0.7723076923076924
[100   5]
Best accuracy is 0.8871794871794871
[100  20]
Best accuracy is 0.8651282051282051
[100 100]
Best accuracy is 0.7964102564102564
{'max_tree_depth': 20, 'min_sample_size': 5}
preprune =  0.825

[20+-5, 5+-2]

[15  2]
Best accuracy is 0.898974358974359
[15  5]
Best accuracy is 0.8958974358974359
[15  7]
Best accuracy is 0.8856410256410256
[20  2]
Best accuracy is 0.897948717948718
[20  5]
Best accuracy is 0.8866666666666667
[20  7]
Best accuracy is 0.8938461538461538
[25  2]
Best accuracy is 0.8938461538461538
[25  5]
Best accuracy is 0.8969230769230769
[25  7]
Best accuracy is 0.8917948717948718
{'max_tree_depth': 15, 'min_sample_size': 2}
preprune =  0.905

[15+-2, 3+-1]

[13  2]
Best accuracy is 0.8953846153846153
[13  3]
Best accuracy is 0.8984615384615384
[13  4]
Best accuracy is 0.8948717948717949
[15  2]
Best accuracy is 0.8984615384615384
[15  3]
Best accuracy is 0.8871794871794871
[15  4]
Best accuracy is 0.9035897435897436
[17  2]
Best accuracy is 0.8907692307692308
[17  3]
Best accuracy is 0.897948717948718
[17  4]
Best accuracy is 0.8958974358974359

results mean = [0.89410256 0.89564103 0.88871795 0.89410256 0.88384615 0.89564103
 0.88897436 0.89230769 0.89564103]

{'max_tree_depth': 13, 'min_sample_size': 3}
preprune =  0.91 <--------------------------------- Optimal

preprune =  0.91
post prune prepruned tree = 0.91
post prune new tree =  0.91

"""