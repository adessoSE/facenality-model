import numpy as np


def calculate_traits_mean(y):
    y_mean = np.mean(y, axis=0)
    return y_mean
