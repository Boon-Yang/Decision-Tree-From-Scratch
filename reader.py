import csv
import numpy as np


def read_from_csv(csv_file_path):
    with open(csv_file_path, 'r') as input_file:
        return_array = []
        reader = csv.reader(input_file)
        for row in reader:
            return_array.append(row)

    return_array = np.asarray(return_array)
    x, y = return_array[:, :-1], return_array[:, -1:]
    return np.asarray(x).astype(np.int), y.squeeze()
