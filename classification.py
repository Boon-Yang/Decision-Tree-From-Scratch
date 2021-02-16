#############################################################################
# 60012: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the fit(), predict() and prune() methods of
# DecisionTreeClassifier. You are free to add any other methods as needed. 
##############################################################################

import tuning
import metrics
import numpy as np
from copy import deepcopy
from numpy.random import default_rng


class DecisionTreeNode(object):
    def __init__(self, col=None, split_value=None):
        self.node = [col, split_value]
        self.count = 0
        self.true = None
        self.false = None

    def get_next_node(self, next_value):
        if next_value:
            return self.true
        else:
            return self.false

    def set_node(self, new_node, new_count=None):
        self.node = new_node
        if not isinstance(new_node, list):
            self.true = None
            self.false = None
            self.count = new_count


class DecisionTreeClassifier(object):
    """ Basic decision tree classifier
    
    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    
    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """

    def __init__(self):
        self.tree = DecisionTreeNode()
        self.max_depth = 0
        self.min_sample_size = 1
        self.max_tree_depth = 100
        self.generator = default_rng(47)
        self.feature_select = False

        self.is_trained = False

    def update_hyperparameters(self, max_tree_depth=100, min_sample_size=1,
                               seed=47, feature_sel=False):
        if not min_sample_size >= 1:
            print("Update failed. must have at least 1 item in the sample")
            return

        self.max_tree_depth = max_tree_depth
        self.min_sample_size = min_sample_size
        self.generator = default_rng(seed)
        self.feature_select = feature_sel

    def __split_dataset(self, x, y, curr_depth, next_value):
        if len(y) <= self.min_sample_size or curr_depth == self.max_tree_depth:
            class_y, full_y, count_y = \
                np.unique(y, return_inverse=True, return_counts=True)
            index_max = count_y.argmax(axis=0)
            count_max = count_y.max(axis=0)
            next_value.set_node(class_y[index_max], count_max)
            if curr_depth > self.max_depth:
                self.max_depth = curr_depth
            return

        if self.feature_select:
            feature_columns = \
                self.generator.choice(x.shape[1], replace=False,
                                      size=np.random.randint(1, x.shape[1] +1))

            best_col, best_split_val = \
                tuning.find_optimal_split(x, y, feature_columns)
        else:
            best_col, best_split_val = \
                tuning.find_optimal_split(x, y, range(x.shape[1]))

        if best_col is None:
            class_y, full_y, count_y = \
                np.unique(y, return_inverse=True, return_counts=True)
            index_max = count_y.argmax(axis=0)
            count_max = count_y.max(axis=0)
            next_value.set_node(class_y[index_max], count_max)
            if curr_depth > self.max_depth:
                self.max_depth = curr_depth
            return

        next_value.set_node([best_col, best_split_val])
        next_value.true = DecisionTreeNode()
        next_value.false = DecisionTreeNode()
        next_true = next_value.get_next_node(True)
        next_false = next_value.get_next_node(False)

        child_1_x, child_1_y, child_2_x, child_2_y = \
            tuning.binary_split(x, y, best_col, best_split_val)

        self.__split_dataset(child_1_x, child_1_y, curr_depth + 1,
                             next_true)
        self.__split_dataset(child_2_x, child_2_y, curr_depth + 1,
                             next_false)

    def fit(self, x, y):
        """ Constructs a decision tree classifier from data
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (N, K) 
                           N is the number of instances
                           K is the number of attributes
        y (numpy.ndarray): Class labels, numpy array of shape (N, )
                           Each element in y is a str 
        """
        
        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."
        
        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################
        self.__split_dataset(x, y, 0, self.tree)

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.
        
        Assumes that the DecisionTreeClassifier has already been trained.
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (M, K) 
                           M is the number of test instances
                           K is the number of attributes
        
        Returns:
        numpy.ndarray: A numpy array of shape (M, ) containing the predicted
                       class label for each instance in x
        """
        
        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")
        
        # set up an empty (M, ) numpy array to store the predicted labels 
        # feel free to change this if needed
        predictions = []
        
        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################
        # remember to change this if you rename the variable
        for instance in x:
            current_node = self.tree
            while current_node.true is not None:
                col, split = current_node.node
                current_node = \
                    current_node.get_next_node(instance[col] <= split)

            predictions.append(current_node.node)

        return np.asarray(predictions)

    def find_leaf_nodes(self, current_node, leaf_nodes, path=[]):
        # Adding to list of leaf nodes
        if not isinstance(current_node.true.node, list) and \
                not isinstance(current_node.false.node, list):
            leaf_nodes.append(path)
            return

        if isinstance(current_node.true.node, list):
            next_path = path.copy()
            next_path.append(True)
            self.find_leaf_nodes(current_node.true, leaf_nodes, next_path)

        if isinstance(current_node.false.node, list):
            next_path = path.copy()
            next_path.append(False)
            self.find_leaf_nodes(current_node.false, leaf_nodes, next_path)

    def prune_leaf_node(self, leaf_node):
        current_node = self.tree

        for pathing in leaf_node:
            current_node = current_node.get_next_node(pathing)

        count_true = current_node.true.count
        count_false = current_node.false.count

        if count_true >= count_false:
            current_node.set_node(current_node.true.node, count_true)
        else:
            current_node.set_node(current_node.false.node, count_false)

    def prune_all(self, leaf_nodes, x_val, y_val,
                  metric=metrics.accuracy):
        changed = False
        for leaf in leaf_nodes:
            new_tree = deepcopy(self)
            new_tree.prune_leaf_node(leaf)

            y_pred_new = new_tree.predict(x_val)
            new_accuracy = metric(y_pred_new, y_val)

            y_pred_old = self.predict(x_val)
            old_accuracy = metric(y_pred_old, y_val)

            if new_accuracy > old_accuracy:
                print(f"Pruned node at: {leaf}")
                self.prune_leaf_node(leaf)
                changed = True

        return changed

    def prune(self, x_val, y_val):
        """ Post-prune your DecisionTreeClassifier given some optional validation dataset.

        You can ignore x_val and y_val if you do not need a validation dataset for pruning.

        Args:
        x_val (numpy.ndarray): Instances of validation dataset, numpy array of shape (L, K).
                           L is the number of validation instances
                           K is the number of attributes
        y_val (numpy.ndarray): Class labels for validation dataset, numpy array of shape (L, )
                           Each element in y is a str 
        """
        
        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")

        #######################################################################
        #                 ** TASK 4.1: COMPLETE THIS METHOD **
        #######################################################################
        current_node = self.tree

        leaf_nodes = []
        self.find_leaf_nodes(current_node, leaf_nodes)

        while self.prune_all(leaf_nodes, x_val, y_val):
            leaf_nodes = []
            self.find_leaf_nodes(current_node, leaf_nodes)

        return
