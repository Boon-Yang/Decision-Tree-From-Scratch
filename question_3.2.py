import reader
import numpy as np
import metrics
from classification import DecisionTreeClassifier


x_train, y_train = reader.read_from_csv("data/train_full.txt")
x_test, y_test = reader.read_from_csv("data/test.txt")

# Trees initialisation
tree_full = DecisionTreeClassifier()

mean_acc, std_dev_acc = \
    metrics.k_cross_val(tree_full, x_train, y_train, k=10, seed=42)

# Q3.2
print('Q3.2')
print(mean_acc, std_dev_acc)

y_pred = tree_full.predict(x_test)

test_accuracy = metrics.accuracy(y_pred, y_test)
test_precision = metrics.precision(y_pred, y_test)
test_recall= metrics.recall(y_pred, y_test)
test_f_score= metrics.f_score(y_pred, y_test)
labels, confusion_matrix = metrics.conf_matrix(y_pred, y_test)

# Q3.3
print('Q3.3')
print(labels)
print(confusion_matrix)
print('Test acc:', test_accuracy)
print('Test precision:', test_precision)
print('Test recall:', test_recall)
print('Test f1 score:', test_f_score)
print('Average test recall:', metrics.avg_recall(y_pred, y_test))
print('Average test precision:', metrics.avg_precision(y_pred, y_test))
print('Average test f1-score:', metrics.avg_f_score(y_pred, y_test))

