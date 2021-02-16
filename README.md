## 60012 Introduction to Machine Learning: Coursework 1 (Decision Trees)

### Introduction

This repository contains the skeleton code and dataset files that you need 
in order to complete the coursework.


### Data

The ``data/`` directory contains the datasets you need for the coursework.

The primary datasets are:
- ``train_full.txt``
- ``train_sub.txt``
- ``train_noisy.txt``
- ``validation.txt``

Some simpler datasets that you may use to help you with implementation or 
debugging:
- ``toy.txt``
- ``simple1.txt``
- ``simple2.txt``

The official test set is ``test.txt``. Please use this dataset sparingly and 
purely to report the results of evaluation. Do not use this to optimise your 
classifier (use ``validation.txt`` for this instead). 


### Main Modules

- ``classification.py``

	* Contains the code for the ``DecisionTreeClassifier`` class. Our task 
was to implement the ``train()``, ``predict()`` and ``prune()`` methods.


- ``improvement.py``

	* Contains the code for the ``train_and_predict()`` function (Task 4.2).
This function acts as an interface to our new/improved decision tree classifier. 
We have implemented a random forest with random features selection and random data 
sampling as improvements. Additionally we also tried out post and pre pruning of the
decision tree.





