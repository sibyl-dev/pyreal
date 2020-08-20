import numpy as np

"""
Finds a set of representative datapoints from a dataset
Includes code from
https://towardsdatascience.com/k-medoids-clustering-on-iris-data-set-1931bf781e05
"""


def sample_points(X, k):
    from numpy.random import choice
    from numpy.random import seed

    seed(1)
    samples = choice(len(X), size=k, replace=False)
    return samples


def compute_d_p(X, medoids_inds, p):
    m = len(X)
    medoids = X[medoids_inds,:]
    medoids_shape = medoids.shape
    # If a 1-D array is provided,
    # it will be reshaped to a single row 2-D array
    if len(medoids_shape) == 1:
        medoids = medoids.reshape((1, len(medoids)))
    k = len(medoids)

    S = np.empty((m, k))

    for i in range(m):
        d_i = np.linalg.norm(X[i, :] - medoids, ord=p, axis=1)
        S[i, :] = d_i ** p

    return S


def assign_labels(S):
    return np.argmin(S, axis=1)


def update_medoids(X, medoids_inds, p):
    S = compute_d_p(X, medoids_inds, p)
    labels = assign_labels(S)

    out_medoids_inds = medoids_inds

    avg_dissimilarity = np.sum(np.min(S,axis=1))
    for i in set(labels):
        new_medoids_inds = medoids_inds.copy()

        cluster_points = np.where(labels == i)[0]
        for point in cluster_points:
            new_medoids_inds[i] = point
            new_dissimilarity = np.sum(np.min(compute_d_p(X, new_medoids_inds, p),axis=1))

            if new_dissimilarity < avg_dissimilarity:
                avg_dissimilarity = new_dissimilarity

                out_medoids_inds[i] = point

    return out_medoids_inds


def has_converged(old_medoids, medoids):
    return set(old_medoids) == set(medoids)


def kmedoids(X, k, p, starting_medoids_inds=None, max_steps=np.inf):
    X = np.asanyarray(X)
    if starting_medoids_inds is None:
        medoids_inds = sample_points(X, k)
    else:
        medoids_inds = starting_medoids_inds

    converged = False
    i = 1
    while (not converged) and (i <= max_steps):
        old_medoids_inds = medoids_inds.copy()

        medoids_inds = update_medoids(X, medoids_inds, p)

        converged = has_converged(old_medoids_inds, medoids_inds)
        i += 1

    S = compute_d_p(X, medoids_inds, p)
    labels = assign_labels(S)
    return (medoids_inds, labels)

