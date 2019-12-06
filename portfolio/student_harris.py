import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def get_interest_points(image, feature_width):
	"""
	JR adds: to ensure compatability with project 4A, you simply need to use
	this function as a wrapper for your 4A code.  Guidelines below left
	for historical reference purposes.

	Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
	You can create additional interest point detector functions (e.g. MSER)
	for extra credit.

	If you're finding spurious interest point detections near the boundaries,
	it is safe to simply suppress the gradients / corners near the edges of
	the image.

	Useful in this function in order to (a) suppress boundary interest
	points (where a feature wouldn't fit entirely in the image, anyway)
	or (b) scale the image filters being used. Or you can ignore it.

	By default you do not need to make scale and orientation invariant
	local features.

	The lecture slides and textbook are a bit vague on how to do the
	non-maximum suppression once you've thresholded the cornerness score.
	You are free to experiment. For example, you could compute connected
	components and take the maximum value within each component.
	Alternatively, you could run a max() operator on each sliding window. You
	could use this to ensure that every interest point is at a local maximum
	of cornerness.

	Args:
	-   image: A numpy array of shape (m,n,c),
				image may be grayscale of color (your choice)
	-   feature_width: integer representing the local feature width in pixels.

	Returns:
	-   x: A numpy array of shape (N,) containing x-coordinates of interest points
	-   y: A numpy array of shape (N,) containing y-coordinates of interest points
	-   confidences (optional): numpy nd-array of dim (N,) containing the strength
			of each interest point
	-   scales (optional): A numpy array of shape (N,) containing the scale at each
			interest point
	-   orientations (optional): A numpy array of shape (N,) containing the orientation
			at each interest point
	"""
	k = 0.04

	img = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if len(image.shape) > 2 else image

	dx, dy = cv.Sobel(img, cv.CV_16S, 1, 0), cv.Sobel(img, cv.CV_16S, 0, 1)
	Ixx, Ixy, Iyy = dx**2, dx*dy, dy**2

	gk = cv.getGaussianKernel(ksize=5, sigma=0)
	blur = lambda img: cv.filter2D(cv.filter2D(img, -1, gk), -1, gk.T)

	Ixx, Ixy, Iyy = blur(Ixx), blur(Ixy), blur(Iyy)

	R = (Ixx*Iyy - Ixy**2) - k * (Ixx + Iyy)**2

	temp = []

	thresh = 0.01 * np.max(R)

	for y in range(R.shape[0]):
		for x in range(R.shape[1]):
			if R[y, x] <= thresh: continue
			temp.append([y, x, R[y, x]])

	temp.sort(key=lambda x: x[-1], reverse=True)
	yarr, xarr = np.array(temp)[:,0], np.array(temp)[:,1]

	radii = []

	newarr = np.zeros((xarr.shape[0], 3))
	newarr[0,:] = np.array([yarr[0], xarr[0], 0.0])

	for i in range(1, xarr.shape[0]):
		y1, x1 = yarr[i], xarr[i]

		dists = np.sqrt((x1-xarr[:i])**2 + (y1-yarr[:i])**2)
		min_i = np.argmin(dists) # minimum distance index

		newarr[i,:] = np.array([yarr[min_i], xarr[min_i], dists[min_i]])

	newarr.view('i8,i8,i8').sort(order=['f2'], axis=0)
	newarr = newarr[::-1]
	newarr = newarr[:2000]

	y_vals, x_vals = newarr[:,0], newarr[:,1]

	return x_vals, y_vals
