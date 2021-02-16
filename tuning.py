import reader
import numpy as np
from numpy.random import default_rng


def entropy(y):
    full_y, counts_full = np.unique(y, return_counts=True)

    total_instances = len(y)

    probabilities = counts_full / total_instances
    log_prob = np.log2(probabilities)

    entropies = np.multiply(probabilities, log_prob)

    entropies = np.sum(entropies, axis=None)

    return entropies * -1


def binary_split(parent_x, parent_y, split_column, split_value):
    col_x = parent_x[:, split_column]

    valid_indices = [index for index, value in enumerate(col_x)
                     if value <= split_value]
    invalid_indices = [index for index, value in enumerate(col_x)
                       if value > split_value]

    child_1_x = parent_x[valid_indices]
    child_1_y = parent_y[valid_indices]

    child_2_x = parent_x[invalid_indices]
    child_2_y = parent_y[invalid_indices]

    return child_1_x, child_1_y, child_2_x, child_2_y


def information_gain(parent_y, children_y):
    parent_entropy = entropy(parent_y)
    total_samples = len(parent_y)
    children_entropy = []

    for child_y in children_y:
        child_entropy = entropy(child_y)
        child_entropy *= len(child_y)/total_samples
        children_entropy.append(child_entropy)

    children_entropy = np.asarray(children_entropy)
    children_entropy = children_entropy.sum(axis=None)

    return parent_entropy - children_entropy


def find_optimal_split(x, y, columns):
    max_info_gain = 0
    max_col = None
    max_split_val = None

    for column in columns:
        max_value = x.max(axis=0)[column]
        min_value = x.min(axis=0)[column]

        for split_value in range(min_value, max_value):
            child_1_x, child_1_y, child_2_x, child_2_y = \
                binary_split(x, y, column, split_value)

            if len(child_2_y) == 0:
                break

            info_gained = information_gain(y, (child_1_y, child_2_y))

            if max_info_gain == 0:
                if info_gained > max_info_gain:
                    max_info_gain = info_gained
                    max_col = column
                    max_split_val = split_value
            else:
                if info_gained >= max_info_gain:
                    max_info_gain = info_gained
                    max_col = column
                    max_split_val = split_value

    return max_col, max_split_val


if __name__ == "__main__":
    toy_x, toy_y = reader.read_from_csv("data/toy.txt")
    print(entropy(toy_y))

    child_1_x, child_1_y, child_2_x, child_2_y = \
        binary_split(toy_x, toy_y, 0, 3)
    print(child_1_y, child_2_y)

    print(entropy(child_1_y), entropy(child_2_y))

    info_gained = information_gain(toy_y, (child_1_y, child_2_y))
    print(info_gained)

    optimal_col, optimal_val = find_optimal_split(toy_x, toy_y,
                                                  range(toy_x.shape[1]))
    print(optimal_col, optimal_val)
