##############################################################################
# 60012: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train_and_predict() function. 
#             You are free to add any other methods as needed. 
##############################################################################

from random_forest import RandomForestClassifier


def train_and_predict(x_train, y_train, x_test, x_val, y_val):
    """ Interface to train and test the new/improved decision tree.
    
    This function is an interface for training and testing the new/improved
    decision tree classifier. 

    x_train and y_train should be used to train your classifier, while 
    x_test should be used to test your classifier. 
    x_val and y_val may optionally be used as the validation dataset. 
    You can just ignore x_val and y_val if you do not need a validation dataset.

    Args:
    x_train (numpy.ndarray): Training instances, numpy array of shape (N, K) 
                       N is the number of instances
                       K is the number of attributes
    y_train (numpy.ndarray): Class labels, numpy array of shape (N, )
                       Each element in y is a str 
    x_test (numpy.ndarray): Test instances, numpy array of shape (M, K) 
                            M is the number of test instances
                            K is the number of attributes
    x_val (numpy.ndarray): Validation instances, numpy array of shape (L, K) 
                       L is the number of validation instances
                       K is the number of attributes
    y_val (numpy.ndarray): Class labels of validation set, numpy array of shape (L, )
    """

    #######################################################################
    #                 ** TASK 4.2: COMPLETE THIS FUNCTION **
    #######################################################################
    # TODO: Train new classifier
    forest = RandomForestClassifier()
    # Forest is trained on the best hyperparameter set
    forest.update_hyperparameters(feature_sel=True, cross_val=False,
                                  max_tree_depth=13, min_sample_size=2,
                                  num_trees=20)

    forest.fit(x_train, y_train)
    # set up an empty (M, ) numpy array to store the predicted labels
    # feel free to change this if needed

    # TODO: Make predictions on x_test using new classifier        
    predictions = forest.predict(x_test)

    # return result on best classifier option
    # remember to change this if you rename the variable
    return predictions


