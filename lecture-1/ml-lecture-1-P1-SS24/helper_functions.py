import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import time
import random
import math
from plot_functions import visualize_scalar_product_2D, get_angle
import random
from PIL import Image, ImageOps


def generate_random_vector(dim=2, use_numpy=False):
    """
    Creates a random vector with homogeneously sampled values in R^N.
    :param dim: the dimension of the vector
    :param seed: a random seed
    :return: A random vector of length dim
    """
    rnd_vec = [0] * dim # TODO: Overwrite this with your implementation.
    if not use_numpy:
        
        for i in range(0,dim):

            rnd_vec[i] = random.random()

    else:
        
        rnd_vec = np.random.rand(dim)

    return rnd_vec


def flatten_matrix(x1):
    """
    Transforms a n_1 x n_2 x ... x n_m matrix into a vector of length n_1 * n_2 * ... * n_m
    :param x1: The matrix
    :return: The flattened vector
    """
    x1 = np.resize(x1, (x1.size))
    return x1


# Helper function to define the inner product.
def inner_prod(x1, x2, use_numpy=False):
    """
    Computes the inner product of two vectors
    :param x1: first vector
    :param x2: second vector
    :return: the inner product
    """
    # assert len(x1) == len(x2), "Error, cannot compute the inner product because the vector lengths are not equal" #TBC later if hv time
    p = 0 # TODO: Overwrite this value with your implementation
    if not use_numpy:
        sum = 0
        for i in range(0, len(x1)):
            sum += x1[i]* x2[i]

        p = sum

    else:
        
        p = np.inner(x1,x2)

    return p


# Helper function to determine the magnitude of a vector
def mag(x, use_numpy=False):
    """
    Computes the magnitude of a vector
    :param x: the vector
    :return: the magnitude of the vector
    """
    m = 0 # TODO: Overwrite this value with your implementation
    if use_numpy is False:
        sum = 0

        for i in range(0, len(x)):
            sum += (x[i])**2

        m = math.sqrt(sum)

    else:
        
        m = np.linalg.norm(x)

    return m



# Helper function to determine the radius of a set of points
def get_radius(D):
    """
    Computes the radius of a point cloud as the distance to the point that is farthest from the centre
    :param D: A vector or list of vectors
    :return: The radius
    """
    max_r = 0
    # <START Your code here>

    # <END Your code here>
    return max_r


def vector_rotate2D(x, deg):
    """
    Computes the rotation of a vector
    :param x: the 2D vector to rotate as a list with 2 elements
    :param deg: the angle in degrees (not rad!) for the CCW rotation
    :return: the rotated 2D vector as a list with 2 elements (the x and y coordinate)
    """

    theta = np.deg2rad(deg)
    x_rotated = [0, 0]
    x_rotated[0] = x[0] * np.cos(theta) - x[1] * np.sin(theta)
    x_rotated[1] = x[0] * np.sin(theta) + x[1] * np.cos(theta)
    return x_rotated


def gen_lin_sep_dataset(n_samples, guarantee_separable=False):
    """
    Generates a dataset of 2D points that are linearly separable in two classes.
    :param n_samples:  Number of points
    :param guarantee_separable: whether to guarantee that the dataset is linearly separable. Note that if this is true,
    it is no longer guaranteed that the function will return the desired number of samples.
    :return: A tuple where the first element is the points in one class and the second is the points in the other class.
    """
    y_multi = 5
    D, c_idx = datasets.make_blobs(n_samples=[n_samples // 2, n_samples // 2], centers=[[2, 0], [-2, 0]],
                                   n_features=2)
    # A manual dataset:
    c_idx = np.array([0,0,0,1,1,1])
    D = np.array([[1, 3],[1.5, -2],[4, 1],[-4, 3],[-2.5, -3],[-1, 1.5]])
    y_multi = 1

    # D = np.round(D, 1)
    # Remove samples that destroy separability if required.
    if guarantee_separable:
        sep_idx = [i for i in range(len(D)) if D[i,0] < 0 and c_idx[i] == 1 or D[i,0] > 0 and c_idx[i] == 0]
        D = D[sep_idx]
        c_idx = c_idx[sep_idx]

    D[:,1] *= y_multi

    C = D[c_idx == 1]
    NotC = D[c_idx == 0]

    return C, NotC


# Normalize a 2d vector
def normalize_2d(vec):
    """
    Normalize a 2d vector
    :param vec: the vector to normalize
    :return: the normalized vector
    """
    normalized_vec = [x / mag(vec) for x in vec]
    return normalized_vec


def project_2D(x1, x2):
    """
    Given two vectors, compute the projection of one vector onto another.
    :param x1: The vector to project onto the other vector
    :param x2: The vector on which to project
    :return: the projected vector
    """
    # Normalize the vector to project onto
    normalized_x2 = normalize_2d(x2)

    # Get the magnitude of the projected vector by computing its inner product with the normalized vector to project onto.
    proj_mag = inner_prod(x1, normalized_x2)

    # Compute the projection
    proj = [x * proj_mag for x in normalized_x2]
    return proj


def plot_inner_product(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)

    inner_prod_magnitude = inner_prod(x1,x2)
    visualize_scalar_product_2D(x1, x2, inner_prod_magnitude, vector_rotate2D)


def load_image(filename="cat.jpg", scale_to_size=None, grayscale=True):
    image = Image.open(filename)
    if scale_to_size is not None:
        image = image.resize(scale_to_size)
    if grayscale:
        image = ImageOps.grayscale(image)
    data = np.asarray(image, dtype=float)
    return data


def show_image(numpy_image, scale_to_width=200):
    image = Image.fromarray((numpy_image * 255).astype(np.uint8))
    # image = Image.fromarray(numpy_image)
    scale_factor = scale_to_width / image.size[0]
    image = ImageOps.scale(image, scale_factor, resample=Image.BOX)
    image.show()

# x1 = generate_random_vector(10, True)
# x1_256 = x1*256
# show_image(x1_256.astype(int),200)

# a = np.random.rand(2)
# show_image(a, 200)


def compute_angle(x1, x2):
    """
    Computes the angle (in degrees) between two vectors.
    :param x1: the first vector
    :param x2: the second vector
    :return: the angle in degrees
    """
    # ang = 45 # TODO: overwrite this value with your implementation
    # <START Your code here>

    x1_x2 = inner_prod(x1,x2)

    ang = math.acos(x1_x2 / (mag(x1) * mag(x2)))

    # <END Your code here>
    return ang

# a = generate_random_vector(2,True)
# print("a: ", a, "\n")
# b = generate_random_vector(2,True)
# print("b: ", b, "\n")
# print("angle: ", compute_angle(a,b))

# x1_20k = generate_random_vector(20000, True)
# x2_20k = generate_random_vector(20000, True)

# x1_20k_r = np.reshape(x1_20k*256, (200,100))
# x2_20k_r = np.reshape(x2_20k*256, (200,100))

# show_image(x1_20k_r.astype(int), 200)
# show_image(x2_20k_r.astype(int), 200)

x1 = load_image(filename="cat.jpg", scale_to_size=(100, 100))
x2 = load_image(filename="dog.jpg", scale_to_size=(100, 100))
show_image(x1)
show_image(x2)

x1f = flatten_matrix(x1)
x2f = flatten_matrix(x2)
print('angle', compute_angle(x1f,x2f))