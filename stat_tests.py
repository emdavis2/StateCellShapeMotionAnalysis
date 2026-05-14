import numpy as np
from scipy.stats import mannwhitneyu


#https://www.geeksforgeeks.org/machine-learning/permutation-tests-in-machine-learning/#what-are-permutation-tests
def permutation_test(group_a, group_b, num_permutations=10000):
    observed_statistic = np.median(group_a) - np.median(group_b)
    combined_data = np.concatenate((group_a, group_b))
    permuted_statistics = []

    for _ in range(num_permutations):
        np.random.shuffle(combined_data)
        perm_group_a = combined_data[:len(group_a)]
        perm_group_b = combined_data[len(group_a):]
        perm_statistic = np.median(perm_group_a) - np.median(perm_group_b)
        permuted_statistics.append(perm_statistic)

    p_value = np.sum(np.abs(permuted_statistics) >= np.abs(observed_statistic)) / num_permutations
    return p_value

    # # Example usage:
    # group_a = np.array([85, 88, 90, 84, 86])
    # group_b = np.array([78, 80, 82, 85, 79])
    # p_value = permutation_test(group_a, group_b)
    # print("p-value:", p_value)