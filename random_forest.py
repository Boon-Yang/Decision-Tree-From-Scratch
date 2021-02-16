import reader
import numpy as np
import metrics
from classification import DecisionTreeClassifier


class RandomForestClassifier(object):
    def __init__(self):
        self.num_trees = 10
        self.cross_val = True
        self.k_fold = 10
        self.trees = []
        self.min_sample_size = 1
        self.max_tree_depth = 100
        self.cross_val = True
        self.feature_select = False

        self.is_trained = False

    def update_hyperparameters(self, num_trees=10, min_sample_size=1,
                               max_tree_depth=100, k_fold=10,
                               feature_sel=None, cross_val=None):
        self.max_tree_depth = max_tree_depth
        self.min_sample_size = min_sample_size
        self.num_trees = num_trees
        self.k_fold = k_fold
        if feature_sel is not None:
            self.feature_select = feature_sel
        if cross_val is not None:
            self.cross_val = cross_val

    def fit(self, x, y):
        for tree in range(self.num_trees):
            new_tree = DecisionTreeClassifier()
            new_tree.update_hyperparameters(self.max_tree_depth,
                                            self.min_sample_size,
                                            seed=tree,
                                            feature_sel=self.feature_select)
            if self.cross_val:
                metrics.k_cross_val(new_tree, x, y, self.k_fold, seed=tree)
            else:
                sample_indices = np.random.choice(len(y),
                                                  size=int(0.8 * x.shape[0]),
                                                  replace=False)
                x_new = x[sample_indices]
                y_new = y[sample_indices]
                new_tree.fit(x_new, y_new)

            self.trees.append(new_tree)

        self.is_trained = True

    def predict(self, x):
        if not self.is_trained:
            print("Classifier is not trained")
            return

        result = []
        predictions = []
        for tree in self.trees:
            prediction = tree.predict(x)
            predictions.append(prediction)

        for col in range(len(x)):
            votes = []
            for prediction_set in predictions:
                votes.append(prediction_set[col])

            votes_class, votes_count = np.unique(votes, return_counts=True)
            result.append(votes_class[np.argmax(votes_count)])

        return np.asarray(result)
