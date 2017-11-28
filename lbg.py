"""
Linde-Buzo-Gray / Generalized Lloyd algorithm implementation in Python *3*.
Herleeyandi Markoni
11/27/2017
"""

import math
import numpy as np
from functools import reduce
from collections import defaultdict

_size_data = 0
_dim = 0

def generate_codebook(data, size_codebook, epsilon=0.00005):
    """
    This function will generate the codebook by given the data and the given size. Epsilon is the threshold taht used 
    during splitting and looping.
    """
    global _size_data, _dim

    _size_data = len(data)
    assert _size_data > 0

    _dim = len(data[0])
    assert _dim > 0

    codebook = []
    codebook_abs = [_size_data]
    codebook_rel = [1.0]

    # get the initial codevector by taking the average vector of whole input data
    c0 = avg_all_vectors(data, _dim, _size_data)
    codebook.append(c0)

    # initial average distortion
    avg_dist = initial_avg_distortion(c0, data)

    # splitting process until we have exactly same number of codevector with the size of codebook.
    while len(codebook) < size_codebook:
        codebook, codebook_abs, codebook_rel, avg_dist = split_codebook(data, codebook,
                                                                        epsilon, avg_dist)
    #return the result
    return codebook, codebook_abs, codebook_rel

def split_codebook(data, codebook, epsilon, initial_avg_dist):
    """
    Split into 2 codebook the get the best centroid as the new codevector.
    """

    # split into 2
    new_cv = []
    for c in codebook:
        # plus and minus epsilon for the new codebook
        c1 = new_codevector(c, epsilon)
        c2 = new_codevector(c, -epsilon)
        new_cv.extend((c1, c2))

    codebook = new_cv
    len_codebook = len(codebook)
    abs_weights = [0] * len_codebook
    rel_weights = [0.0] * len_codebook

    # Get the best centroid by taking average distortion as cost function. This problems mimic K-Means.
    avg_dist = 0
    err = epsilon + 1
    num_iter = 0
    while err > epsilon:
        # Get nearest codevector.
        closest_c_list = [None] * _size_data    # nearest codevector
        vecs_near_c = defaultdict(list)         # input data vector mapping
        vec_idxs_near_c = defaultdict(list)     # input data index mapping
        for i, vec in enumerate(data):  # for each input vector
            min_dist = None
            closest_c_index = None
            for i_c, c in enumerate(codebook):
                d = get_mse(vec, c)
                # Get the nearest ones.
                if min_dist is None or d < min_dist:
                    min_dist = d
                    closest_c_list[i] = c
                    closest_c_index = i_c
            vecs_near_c[closest_c_index].append(vec)
            vec_idxs_near_c[closest_c_index].append(i)

        # Update the codebook
        for i_c in range(len_codebook):
            vecs = vecs_near_c.get(i_c) or []
            num_vecs_near_c = len(vecs)
            if num_vecs_near_c > 0:
                # assign as new center
                new_c = avg_all_vectors(vecs, _dim)
                codebook[i_c] = new_c
                for i in vec_idxs_near_c[i_c]:
                    closest_c_list[i] = new_c

                # update the weights
                abs_weights[i_c] = num_vecs_near_c
                rel_weights[i_c] = num_vecs_near_c / _size_data

        # Recalculate average distortion
        prev_avg_dist = avg_dist if avg_dist > 0 else initial_avg_dist
        avg_dist = avg_codevector_dist(closest_c_list, data)

        # Recalculate the new error value
        err = (prev_avg_dist - avg_dist) / prev_avg_dist
        num_iter += 1

    return codebook, abs_weights, rel_weights, avg_dist

def avg_all_vectors(vecs, dim=None, size=None):
    """
    This function will get the average of whole data.
    """
    size = size or len(vecs)
    nvec = np.array(vecs)
    nvec = nvec / size
    navg = np.sum(nvec, axis=0)
    return navg.tolist()

def new_codevector(c, e):
    """
    This function will create a new codevector when we split into two.
    """
    nc = np.array(c)
    return (nc * (1.0 + e)).tolist()

def initial_avg_distortion(c0, data, size=None):
    """
    This function will calculate the average distortion of a vector to the input list of vectors.
    """
    size = size or _size_data
    nc = np.array(c0)
    nd = np.array(data)
    f = np.sum(((nc-nd)**2)/size)
    return f

def avg_codevector_dist(c_list, data, size=None):
    """
    This function will calculate the average distortion between list of vector and the input data.
    """
    size = size or _size_data
    nc = np.array(c_list)
    nd = np.array(data)
    f = np.sum(((nc-nd)**2)/size)
    return f

def get_mse(a, b):
    """
    This function will get the squared error, the mean will be calculate later.
    """
    na = np.array(a)
    nb = np.array(b)
    return np.sum((na-nb)**2)