##############################################################################
# 60012: Introduction to Machine Learning
# Coursework 1 example execution code
# Prepared by: Josiah Wang
##############################################################################

import numpy as np

from classification import DecisionTreeClassifier
from improvement import train_and_predict

if __name__ == "__main__":
    print("Loading the training dataset...")
    x = np.array([
            [5,7,1],
            [4,6,2],
            [4,6,3], 
            [1,3,1], 
            [2,1,2], 
            [5,2,6]
        ])
    
    y = np.array(["A", "A", "A", "C", "C", "C"])
    
    print("Training the decision tree...")
    classifier = DecisionTreeClassifier()
    classifier.fit(x, y)

    print("Loading the test set...")
    
    x_test = np.array([
            [1,6,3], 
            [0,5,5], 
            [1,5,0], 
            [2,4,2]
        ])
    
    y_test = np.array(["A", "A", "C", "C"])
    
    print("Making predictions on the test set...")
    predictions = classifier.predict(x_test)
    print("Predictions: {}".format(predictions))
    
    classes = ["A", "C"];
    
    print("Pruning the decision tree...")
    x_val = np.array([
                [6,7,2],
                [3,1,3]
            ])
    y_val = np.array(["A", "C"])
                   
    classifier.prune(x_val, y_val)
    
    print("Making predictions on the test set using the pruned decision tree...")
    predictions = classifier.predict(x_test)
    print("Predictions: {}".format(predictions))

    print("Making predictions on the test set using the improved decision tree...")
    predictions = train_and_predict(x, y, x_test, x_val, y_val)
    print("Predictions: {}".format(predictions))
    
