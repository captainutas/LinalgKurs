import numpy as np
import scipy as sp
from PIL import Image

from collections.abc import Callable
type array = np.ndarray


def viewdigit(
        digit_vec: array
    ) -> None:
    """
    Input: a vector of length 784
    Show a 28x28 image created from the vector
    """
    raw_mat = np.reshape(digit_vec, (28,28))
    # Scale so that the lowest values become 0 and the highest become 256
    scaled_mat = (raw_mat - np.min(raw_mat)*np.ones(raw_mat.shape)) \
        *(256/np.max(raw_mat))  
    img = Image.fromarray(scaled_mat)
    img.load()
    img.show()


def classify_digit(
        digit: array, 
        dist_func: Callable[[array], array]
    ) -> int: 

    dists = dist_func(digit)
    return int(np.argmin(dists))


def label_data(
        input_digits: array,
        classifier_func: Callable[[array], int]
    ) -> array:
    res = np.zeros(input_digits.shape[0], dtype="int")
    for i, digit in enumerate(input_digits):
        res[i] = classifier_func(digit)
    return res


def calc_success_rates(
        test_sets: list[array], 
        label_func: Callable[[array], array]
    ) -> array:

    res = np.zeros(10)
    for i in range(10):
        test_digits = test_sets[i]
        labeled_digits = label_func(test_digits)
        n_success = list(labeled_digits).count(i)
        res[i] = n_success / test_digits.shape[0]
    return res


def get_test_and_train_sets(
        mnist_data: dict
    ) -> tuple[list[array], list[array]]:
    """
    Input: mnist_data, a dictionary containing traningdata and testdata for each integer
    Output:
        test_sets (list[array]). test_sets[n] contains the test set for integer n
        train_sets (list[array]). train_sets[n] contains the training set for integer n
    """
    test_sets = []
    train_sets = []
    for i in range(10):
        test_sets.append(mnist_data.get('test' + str(i)))
        train_sets.append(mnist_data.get('train' + str(i)))
    return test_sets, train_sets



## Part 1 : Centoroid Method


def get_image_mean_distances(
        input_image: array, 
        mean_train_digits: list[array]
    ) -> array:
    """
    
    """
    dists = np.zeros(10)
    for i in range(10):
        dists[i] = np.linalg.norm(input_image - mean_train_digits[i])
    return dists

def classify_using_mean_distances(
        digit: array, 
        mean_train_digits: list[array]
    ) -> int: 

    dist_func = lambda digit: get_image_mean_distances(digit, mean_train_digits)
    return classify_digit(digit, dist_func)


def label_data_using_mean_distances(
        input_digits: array, 
        mean_train_digits: list[array]
    ) -> array:

    classifier_func = lambda digit: classify_using_mean_distances(digit, mean_train_digits)
    return label_data(input_digits, classifier_func)


def calc_success_rates_using_mean_distances(
        test_sets: list[array], 
        mean_train_digits: list[array]
    ) -> array:

    label_func = lambda digits : label_data_using_mean_distances(digits, mean_train_digits)
    res = calc_success_rates(test_sets, label_func)
    return res



## Part 2 : PCA

def get_squared_singular_vectors(train_set: array, n_vectors: int):
    """
    Returns a 784 by n_vectors array with the columns being the first n singular vectors for the transpose of the test set
    """

    # u_mat = np.linalg.svd(train_set.transpose()).U
    # u_mat = u_mat[:,:n_vectors]

    # Found a more efficent way to calculate the singular vectors needed in the scipy.sparse package
    u_mat, _, _ = sp.sparse.linalg.svds(train_set.transpose(), k=n_vectors, return_singular_vectors="u") 
    squared_umat = u_mat @ u_mat.transpose()
    return squared_umat


def get_least_square_distances(digit: array, squared_us: list[array]):
    dists = np.zeros(10)
    for i in range(10):
        dists[i] = np.linalg.norm(digit - (squared_us[i] @ digit)) # Most of the programs time is spent here currently.
    return dists


def classify_using_PCA(digit, squared_us):

    dist_func = lambda digit: get_least_square_distances(digit, squared_us)
    return classify_digit(digit, dist_func)


def label_data_using_PCA(input_digits, squared_us):

    classifier_func = lambda digit: classify_using_PCA(digit, squared_us)
    return label_data(input_digits, classifier_func)


def calc_success_rates_using_PCA(test_sets, squared_us):

    label_func = lambda test_set: label_data_using_PCA(test_set, squared_us)
    res = calc_success_rates(test_sets, label_func)
    return res



def print_results(succes_rates):
    for i, rate in enumerate(succes_rates):
        print(f"{i}: {rate:.2f}%")



def main():

    mnist_data = sp.io.loadmat("mnistdata.mat")
    test_sets, train_sets = get_test_and_train_sets(mnist_data)
    
    print("Calculating mean digits")
    mean_train_digits = [np.mean(train_set, axis=0) for train_set in train_sets]
    print("Calculating mean distance successrate")
    mean_distance_succes_rates = calc_success_rates_using_mean_distances(test_sets, mean_train_digits)

    print("Calculating singular vectors")
    squared_us = [get_squared_singular_vectors(train_set, 5) for train_set in train_sets]
    print("Calculating PCA successrate")
    PCA_succes_rate = calc_success_rates_using_PCA(test_sets, squared_us)

    print("Mean distance results: ")
    print_results(mean_distance_succes_rates)
    print("PCA results: ")
    print_results(PCA_succes_rate)
    
    

if __name__ == "__main__":
    main()

