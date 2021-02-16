import numpy as np
from numpy.random import default_rng


def accuracy(pred, ground_truth):
    if not len(pred) == len(ground_truth):
        print("Predictions must be equal to number of ground truth")
        return None
    if len(pred) == 0:
        print("No predictions and no ground truth")
        return None

    correct = [value for index, value in enumerate(pred)
               if value == ground_truth[index]]
    total = len(pred)

    return len(correct)/total


def split_k_fold(x, k=10, seed=100):
    rg = default_rng(seed)
    indices = rg.permutation(x.shape[0])
    fold_size = x.shape[0] // k

    folds = []

    for times in range(k):
        fold_index = indices[times * fold_size: (times + 1) * fold_size]
        folds.append(fold_index)

    fold_no = 0
    for extra_index in range(fold_size * k, x.shape[0]):
        folds[fold_no].append(indices[extra_index])
        fold_no += 1

    return folds


# splits the training set to get the kth fold
def get_kth_train_test(folds, k, x, y):
    train_indices = [folds[index] for index, value in enumerate(folds) if
                     not index == k]
    test_indices = folds[k]

    train_indices = np.concatenate(train_indices, axis=0)

    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]

    return x_train, y_train, x_test, y_test


# trains tree on the kth fold and returns the accuracy
def train_and_eval_kth(classifier, folds, x, y, k):
    x_train, y_train, x_test, y_test = \
        get_kth_train_test(folds, k, x, y)

    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    return accuracy(prediction, y_test)


def k_cross_val(classifier, x, y, k=10, seed=100):
    results = []
    folds = split_k_fold(x, k, seed)
    for time in range(k):
        result = train_and_eval_kth(classifier, folds, x, y, time)
        results.append(result)

    results = np.array(results)
    mean = np.mean(results)
    standard_dev = np.std(results)
    best_iteration = np.argmax(results)

    best_acc = train_and_eval_kth(classifier, folds, x, y, k=best_iteration)
    print(f"Best accuracy is {best_acc}")

    return mean, standard_dev


def conf_matrix(predictions, ground_truth):
    combined = np.concatenate((ground_truth, predictions))
    class_labels, inverse = np.unique(combined, return_inverse=True)
    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)
    gt = inverse[:len(ground_truth)]
    pred = inverse[len(predictions):]

    for index in range(len(pred)):
        confusion[gt[index]][pred[index]] += 1

    return class_labels, confusion


def precision(predictions, ground_truth):
    classes, confusion = conf_matrix(predictions, ground_truth)
    total_pos_pred = np.sum(confusion, axis=0)
    true_pos = [value[index] for index, value in enumerate(confusion)]

    return true_pos / total_pos_pred


def recall(predictions, ground_truth):
    classes, confusion = conf_matrix(predictions, ground_truth)
    total_pos = np.sum(confusion, axis=1)
    true_pos = [value[index] for index, value in enumerate(confusion)]

    return true_pos / total_pos


def f_score(predictions, ground_truth, beta=1):
    prec = precision(predictions, ground_truth)
    rec = recall(predictions, ground_truth)

    f = (1 + beta**2) * prec * rec / (beta**2 * prec + rec)

    return f


def avg_precision(predictions, ground_truth):
    p = precision(predictions, ground_truth)
    return np.average(p)


def avg_recall(predictions, ground_truth):
    r = recall(predictions, ground_truth)
    return np.average(r)


def avg_f_score(predictions, ground_truth, beta=1):
    f1 = f_score(predictions, ground_truth, beta)
    return np.average(f1)


def grid_search(classifier, x_train, y_train, x_val, y_val,
                parameter_space_dict):
    keys = []
    space = []
    for key in parameter_space_dict:
        try:
            eval("classifier." + key)
            keys.append(key)
            space.append(parameter_space_dict[key])
        except NameError:
            print(f"{key} is not in the classifier hyperparameters")
            continue

    combinations = 1
    for parameter in space:
        combinations *= len(parameter)

    grid = np.zeros((combinations, len(keys)), dtype=np.int)

    induce_grid(grid, 0, options=space)

    results = []
    for row in grid:
        for col in range(len(row)):
            row[col] = space[col][row[col]]
            setattr(classifier, keys[col], row[col])
        print(row)
        classifier.fit(x_train, y_train)
        pred = classifier.predict(x_val)
        result = accuracy(pred, y_val)
        print(result)
        results.append(result)

    results = np.array(results)
    best_parameter_set = results.argmax()
    best_parameter_set = grid[best_parameter_set]

    for col in range(len(best_parameter_set)):
        setattr(classifier, keys[col], best_parameter_set[col])

    classifier.fit(x_train, y_train)

    zip_obj = zip(keys, best_parameter_set)
    return dict(zip_obj)


def induce_grid(grid_section, parameter_index, options):
    rows = grid_section.shape[0]
    row_start = 0
    num_options = len(options[parameter_index])
    for option_index in range(num_options):
        grid_section[row_start:row_start + rows // num_options,
                     parameter_index] += option_index
        if not parameter_index == grid_section.shape[1] - 1:
            induce_grid(grid_section[row_start:row_start +
                                     rows // num_options],
                        parameter_index + 1,
                        options)
        row_start += rows // num_options
