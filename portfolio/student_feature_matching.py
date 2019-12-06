import numpy as np
from numpy import linalg as LA

def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    For extra credit you can implement various forms of spatial/geometric
    verification of matches, e.g. using the x and y locations of the features.

    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """
    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################

    confidenceMatches = []

    for i in range(len(features1)):
        dist = np.sqrt(np.sum((features2 - features1[i,:])**2, axis=1))

        min_dist_i = np.argmin(dist)
        bestDist = dist[min_dist_i]

        dist[min_dist_i] = float('inf')

        min_dist_i2 = np.argmin(dist)
        secondBestDist = dist[min_dist_i2]

        ratio = bestDist/secondBestDist
        confidenceMatches.append((ratio, i, min_dist_i))


    confidenceMatches.sort()

    maxFeaturesToReturn = 100 #??

    temp = []
    x = 0

    while len(temp) < maxFeaturesToReturn and x < len(confidenceMatches):
        dontAppend = False
        for y in range(len(temp)):
            if temp[y][2] == confidenceMatches[x][2]:
                dontAppend = True
        if not dontAppend:
            temp.append(confidenceMatches[x])
        x += 1

    temp = np.array(temp)

    confidences, im1, im2 = temp.T

    im1 = np.reshape(im1, (im1.shape[0],1))
    im2 = np.reshape(im2, (im2.shape[0],1))

    matches = np.hstack((im1, im2))
    matches = matches.astype(np.int64)


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return matches, confidences
