import numpy as np
from reader import read_from_csv

# Q 1.1
print("Question 1.1")

# Load dataset

full_x, full_y = read_from_csv("data/train_full.txt")
sub_x, sub_y = read_from_csv("data/train_sub.txt")

# Find shape of dataset

print("Shape of full_x:")
print(full_x.shape)
print("Shape of sub_x:")
print(sub_x.shape)  # same number of attributes

classes_full, full_y, counts_full = np.unique(full_y, return_inverse=True,
                                              return_counts=True)
classes_sub, sub_y, counts_sub = np.unique(sub_y, return_inverse=True,
                                           return_counts=True)

print("Classes of full followed by counts")
print(classes_full, classes_sub)
print("Classes of sub followed by counts")
print(counts_full, counts_sub)  # same set of classes
""" Significantly less G than others, low numbers of Q also in the sub 
    Main has similar numbers of everything, differing by max approx 11%
"""

# Find max and min of all attributes

max_x = full_x.max(axis=0)
min_x = full_x.min(axis=0)
max_sub_x = sub_x.max(axis=0)
min_sub_x = sub_x.min(axis=0)
print("")

# Q 1.2
print("Question 1.2")
print("Max and min values of full and sub set")
print(f"max_x: {max_x}, min_x: {min_x} for full set")
print(f"max_x: {max_sub_x}, min_x: {min_sub_x} for sub set")  # looks like discrete integers
print("")

# Q 1.3
print("Question 1.3")

random_x, random_y = read_from_csv("data/train_noisy.txt")
classes_rand, rand_y, counts_rand = \
    np.unique(random_y, return_inverse=True, return_counts=True)

print("Classes and counts for corrupted set for noisy dataset")
print(classes_rand)
print(counts_rand)
print("Diff between counts in full dataset and noisy dataset")
diff = counts_full - counts_rand
print(classes_rand)
print(diff)

positives = 0
for value in diff:
    if value >= 0:
        positives += value

print("Number of changed items, total followed by proportion")
print(positives)
print(positives/len(full_y))