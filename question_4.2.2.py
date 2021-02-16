import reader
import numpy as np
from random_forest import RandomForestClassifier
import metrics

x_train, y_train = reader.read_from_csv("data/train_full.txt")
x_val, y_val = reader.read_from_csv("data/validation.txt")
x_test, y_test = reader.read_from_csv("data/test.txt")

forest = RandomForestClassifier()

"""
# 4.2.3 starts here
forest.update_hyperparameters(num_trees=10, k_fold=10, cross_val=True)
forest.fit(x_train, y_train)

pred_val = forest.predict(x_val)
acc_val_1 = metrics.accuracy(pred_val, y_val)

print(f"10 tree Random Forest Validation Accuracy, ",
      f"without feature selection or sampling: {acc_val_1}")

forest.update_hyperparameters(feature_sel=True, cross_val=True)
forest.fit(x_train, y_train)
pred_val = forest.predict(x_val)
acc_val_2 = metrics.accuracy(pred_val, y_val)

print(f"10 tree Random Forest Validation Accuracy, ",
      f"with feature selection: {acc_val_2}")

forest.update_hyperparameters(feature_sel=False, cross_val=False)
forest.fit(x_train, y_train)
pred_val = forest.predict(x_val)
acc_val_3 = metrics.accuracy(pred_val, y_val)

print(f"10 tree Random Forest Validation Accuracy, ",
      f"without feature selection and with sampling: {acc_val_3}")

forest.update_hyperparameters(feature_sel=True, cross_val=False)
forest.fit(x_train, y_train)
pred_val = forest.predict(x_val)
acc_val_4 = metrics.accuracy(pred_val, y_val)

print(f"10 tree Random Forest Validation Accuracy, ",
      f"with feature selection and sampling: {acc_val_4}")

results = np.asarray([acc_val_1, acc_val_2, acc_val_3, acc_val_4])
print(f"Best result was achieved with setup {np.argmax(results) + 1}")

"""
# 4.2.4 starts here
# Trees start following the best tree model from improvement 1
forest.update_hyperparameters(feature_sel=True, cross_val=False,
                              max_tree_depth=13, min_sample_size=3,
                              num_trees=10)

forest.fit(x_train, y_train)
pred_val = forest.predict(x_val)
acc_val_5 = metrics.accuracy(pred_val, y_val)

print(f"10 Best Tree Random Forest Validation Accuracy, ",
      f"with feature selection and sampling: {acc_val_5}")
      
# start by tuning the trees used
param_space = {"max_tree_depth": [x for x in range(13, 15)],
               "min_sample_size": [y for y in range(2, 4)],
               "num_trees": [10, 20]}
best_param = metrics.grid_search(forest, x_train, y_train, x_val, y_val,
                                 parameter_space_dict=param_space)

print(best_param)

pred_val = forest.predict(x_val)
acc_val_6 = metrics.accuracy(pred_val, y_val)

print(f"Post grid search Random Forest Validation Accuracy, ",
      f"with feature selection and sampling: {acc_val_6}")


"""
10 trees without selection and sampling: 0.92
10 trees with selection and without sampling: 0.96

10 trees without selection and with sampling: 0.95
10 trees with selection and sampling: 0.95

10 trees with selection and cross val: 0.95, 0.95
10 trees with selection and sampling: 0.97, 0.955

10 trees with the best individual tree from grid search, (13, 2): 0.97
10 trees with best forest from grid search, (14, 3): 0.97

20 trees with the best individual tree from grid search, (13, 2): 0.99
20 trees with best forest from grid, (14, 3): 0.97
"""
