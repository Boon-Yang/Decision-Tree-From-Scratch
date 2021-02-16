import reader
import metrics
import numpy as np
from classification import DecisionTreeClassifier


def eval_classifier(classifier, x, y):
    y_pred = classifier.predict(x)

    conf = metrics.conf_matrix(y_pred, y)
    accuracy = metrics.accuracy(y_pred, y)
    precision = metrics.precision(y_pred, y)
    recall = metrics.recall(y_pred, y)
    f1_score = metrics.f_score(y_pred, y, beta=1)
    avg_prec = np.mean(precision)
    avg_rec = np.mean(recall)
    avg_f1 = np.mean(f1_score)

    print("Confusion Matrix: ")
    print(conf)
    print("Accuracy:")
    print(accuracy)
    print("Precision:")
    print(precision)
    print(f"Average Precision: {avg_prec}")
    print("Recall:")
    print(recall)
    print(f"Average Recall: {avg_rec}")
    print("F1_score:")
    print(f1_score)
    print(f"Average F1 Score: {avg_f1}")


x_train, y_train = reader.read_from_csv("data/train_full.txt")
x_sub, y_sub = reader.read_from_csv("data/train_sub.txt")
x_noisy, y_noisy = reader.read_from_csv("data/train_noisy.txt")
x_test, y_test = reader.read_from_csv("data/test.txt")

# Trees initialisation
tree_full = DecisionTreeClassifier()
tree_sub = DecisionTreeClassifier()
tree_noisy = DecisionTreeClassifier()

# Fitting trees
tree_full.fit(x_train, y_train)
tree_sub.fit(x_sub, y_sub)
tree_noisy.fit(x_noisy, y_noisy)

# Evaluation
eval_classifier(tree_full, x_test, y_test)
eval_classifier(tree_sub, x_test, y_test)
eval_classifier(tree_noisy, x_test, y_test)

