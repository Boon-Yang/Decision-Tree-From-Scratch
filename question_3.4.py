import reader
import numpy as np
from scipy import stats
import metrics
from classification import DecisionTreeClassifier

x_train, y_train = reader.read_from_csv("data/train_full.txt")
x_test, y_test = reader.read_from_csv("data/test.txt")

k = 10
seed = 42
folds = metrics.split_k_fold(x_train, k, seed)
trees = []  # Holds all k trees trained using train sets selected by folds

# Train trees

for fold in range(k):

    # Create a new tree, train it and store it in trees collection
    new_tree = DecisionTreeClassifier()
    new_tree_accuracy = metrics.train_and_eval_kth(new_tree, folds,
                                                   x_train, y_train, fold)
    trees.append(new_tree)

    # Print out the result for this fold
    print("tree accuracy (validation)", fold, " = ",
          new_tree_accuracy)
    print("tree accuracy (test)", fold, " = ",
          metrics.accuracy(new_tree.predict(x_test), y_test))

# Find predictions for each folds

predictions = []
for tree in trees:
    prediction = tree.predict(x_test)
    predictions.append(prediction)

# Combine the predictions and get mode

predictions = np.array(predictions)
mode = stats.mode(predictions)[0].flatten()
# print('mode = ', mode)
test_accuracy = metrics.accuracy(mode, y_test)
test_precision = metrics.precision(mode, y_test)
test_recall= metrics.recall(mode, y_test)
test_f_score= metrics.f_score(mode, y_test)
labels, confusion_matrix = metrics.conf_matrix(mode, y_test)

print('Q3.4')
print(labels)
print(confusion_matrix)
print('Test acc:', test_accuracy)
print('Test precision:', test_precision)
print('Test recall:', test_recall)
print('Test f1 score:', test_f_score)
print('Average test recall:', metrics.avg_recall(mode, y_test))
print('Average test precision:', metrics.avg_precision(mode, y_test))
print('Average test f1-score:', metrics.avg_f_score(mode, y_test))

print('mode accuracy = ', metrics.accuracy(mode, y_test))

