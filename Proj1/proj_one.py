import numpy as np
import scipy as sp
from PIL import Image

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


def get_image_distances(
        input_image: array, 
        mean_train_digits: list[array]
    ) -> array:
    """
    
    """
    dists = np.zeros(10)
    for i in range(10):
        dists[i] = np.linalg.norm(input_image - mean_train_digits[i])
    return dists


def classify_digit(
        digit: array, 
        dist_func
    ) -> int: 

    dists = dist_func(digit)
    return int(np.argmin(dists))

def classify_from_distance(
        digit: array, 
        mean_train_digits: list[array]
    ) -> int: 

    dist_func = lambda digit: get_image_distances(digit, mean_train_digits)
    return classify_digit(digit, dist_func)


def label_data(
        input_digits: array,
        classifier_func
    ) -> array:
    res = np.zeros(input_digits.shape[0], dtype="int")
    for i, digit in enumerate(input_digits):
        res[i] = classifier_func(digit)
    return res


def label_data_using_distances(
        input_digits: array, 
        mean_train_digits: list[array]
    ) -> array:

    classifier_func = lambda digit: classify_from_distance(digit, mean_train_digits)
    return label_data(input_digits, classifier_func)





def calc_success_rate_using_distances(
        test_sets: list[array], 
        mean_train_digits: list[array]
    ) -> array:

    res = np.zeros(10)
    for i in range(10):
        test_digits = test_sets[i]
        labeled_digits = label_data_using_distances(test_digits, mean_train_digits)
        n_success = list(labeled_digits).count(i)
        res[i] = n_success / test_digits.shape[0]
    return res



## PCA

def get_singular_vectors(train_set: array, n_vectors):
    """
    Returns a 784 by n_vectors array with the columns being the first n singular vectors for the transpose of the test set
    """
    us = np.linalg.svd(train_set.transpose()).U
    return us[:,:n_vectors]


def get_least_square_distances(digit: array, us: list[array]):
    dists = np.zeros(10)
    for i in range(10):
        dists[i] = np.linalg.norm(digit - ((us[i] @ us[i].transpose()) @ digit))
    return dists


def classify_from_PCA(digit, us):

    dist_func = lambda digit: get_least_square_distances(digit, us)
    return classify_digit(digit, dist_func)


def label_data_using_PCA(input_digits, us):

    classifier_func = lambda digit: classify_from_PCA(digit, us)
    return label_data(input_digits, classifier_func)



def main():
    mnist_data = sp.io.loadmat("mnistdata.mat")
    test_sets, train_sets = get_test_and_train_sets(mnist_data)
    mean_train_digits = [np.mean(train_set, axis=0) for train_set in train_sets]
    #print(calc_success_rate_using_distances(test_sets, mean_train_digits))
    us = [get_singular_vectors(train_set, 5) for train_set in train_sets]
    test = get_least_square_distances(test_sets[0][0], us)
    print(test)
    
    

if __name__ == "__main__":
    main()

