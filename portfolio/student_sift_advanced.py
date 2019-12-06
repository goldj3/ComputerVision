import numpy as np

import numpy as np
import cv2


def get_features(image, x, y, feature_width, scales=None):
    """
    JR Writes: To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    maximal points you may need to implement a more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)


    Below for advanced implementation:

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    gray = np.float64(image)
    blurred = cv2.GaussianBlur(gray,(5,5), 0)
    SobelX = np.array([-1,0,1,-2,0,2,-1,0,1]).reshape(3,3)
    SobelY = np.array([-1,-2,-1,0,0,0,1,2,1]).reshape(3,3)
    Gx = cv2.filter2D(blurred, -1, SobelX)
    Gy = cv2.filter2D(blurred, -1, SobelY)
    theta = np.arctan2(Gy, Gx)
    fv = []
    pad_val = int(feature_width/2)
    paddedTheta = np.pad(theta, ((pad_val,pad_val),(pad_val,pad_val)), 'constant', constant_values=0)
    for i in range(len(x)):
        curX,curY = int(x[i]+pad_val), int(y[i]+pad_val)
        allSubregions = fourByFourSubs(paddedTheta,curX,curY)
        h = []
        for cur in allSubregions:
            h.append(np.histogram(cur, 16, (-np.pi,(np.pi)))[0])
        fv.append(np.array(h))

    fv = np.array(fv)
    fv = fv.reshape((fv.shape[0], fv.shape[1]*fv.shape[2]))

    #fv = normalize(fv)
    #fv = np.clip(fv,0, 0.2)
    #fv = normalize(fv)

    # Empirical studies have demonstrated that this fractional
    # power value maximizes performance on our dataset
    fv = fv**0.864534875237854787576

    return fv


def normalize(vect):
    newarr = np.zeros(vect.shape).astype(np.float)

    for i in range(vect.shape[0]):
        maxv, minv = np.max(vect[i,:]), np.min(vect[i,:])
        newarr[i,:] = ((vect[i,:] - minv) / (maxv - minv))

    return newarr


def fourByFourSubs(paddedImage, curX, curY):
    """
    given a padded image, and an interest point, returns a 16x16 window around the point in 4x4 subregions
    """
    quad1_1 = paddedImage[curY-8:curY-4, curX-8:curX-4]
    quad1_2 = paddedImage[curY-8:curY-4, curX-4:curX]
    quad1_3 = paddedImage[curY-8:curY-4, curX:curX+4]
    quad1_4 = paddedImage[curY-8:curY-4, curX+4:curX+8]

    quad2_1 = paddedImage[curY-4:curY, curX-8:curX-4]
    quad2_2 = paddedImage[curY-4:curY, curX-4:curX]
    quad2_3 = paddedImage[curY-4:curY, curX:curX+4]
    quad2_4 = paddedImage[curY-4:curY, curX+4:curX+8]

    quad3_1 = paddedImage[curY:curY+4, curX-8:curX-4]
    quad3_2 = paddedImage[curY:curY+4, curX-4:curX]
    quad3_3 = paddedImage[curY:curY+4, curX:curX+4]
    quad3_4 = paddedImage[curY:curY+4, curX+4:curX+8]

    quad4_1 = paddedImage[curY+4:curY+8, curX-8:curX-4]
    quad4_2 = paddedImage[curY+4:curY+8, curX-4:curX]
    quad4_3 = paddedImage[curY+4:curY+8, curX:curX+4]
    quad4_4 = paddedImage[curY+4:curY+8, curX+4:curX+8]

    return np.array((quad1_1,quad1_2,quad1_3,quad1_4,
            quad2_1,quad2_2,quad2_3,quad2_4,
            quad3_1,quad3_2,quad3_3,quad3_4,
            quad4_1,quad4_2,quad4_3,quad4_4))
