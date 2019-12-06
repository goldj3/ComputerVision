# Justin Gold
# CSC-483 Project 1
# 9/21/19

#project1.py
import numpy as np
import matplotlib.pyplot as plt
import cv2

def loadppm(filename):
    
    '''Given a filename, return a numpy array containing the ppm image
    input: a filename to a valid ascii ppm file 
    output: a properly formatted 3d numpy array containing a separate 2d array 
            for each colors
    notes: be sure you test for the correct P3 header and use the dimensions and depth 
            data from the header
            your code should also discard comment lines that begin with #
    '''
    file = open(filename, "r")
    allLines = file.read().split() # split every value in file into comma separated list
    dimensions = [int(i) for i in allLines[1:3]]    #[0] = cols, [1] = rows
    vals = allLines[4:]
    vals = [int(i) for i in vals]
    red = np.array([vals[i] for i in range(0, len(vals), 3)], dtype=np.uint8) # load every 1st triplet value into red array w/ uint8 datatype
    green = np.array([vals[i] for i in range(1, len(vals), 3)], dtype=np.uint8) # load every 2nd triplet value into green array w/ uint8 datatype
    blue = np.array([vals[i] for i in range(2, len(vals), 3)],dtype=np.uint8) # load every 3rd triplet value into blue array w/ uint8 datatype
    red = red.reshape(dimensions[1], dimensions[0])  # reshape to match dimensions of original pic (ppm has (col, row), np has (row, col))
    green = green.reshape(dimensions[1], dimensions[0])
    blue = blue.reshape(dimensions[1], dimensions[0])
    return np.dstack([red,green,blue]) #



def GetGreenPixels(img):
    '''given a numpy 3d array containing an image, return the green channel'''
    green = []
    for i in range(0, len(img)): # gets every 2d array in the 3d array
        for j in range(0, len(img[0])): # gets every 1d array (which represents one RGB triplet) in each 2d array
            green.append(img[i][j][1])  # gets the green value from each RGB triplet and appends to new list
    return np.array(green).reshape(img.shape[0],img.shape[1])

def GetBluePixels(img):
    '''given a numpy 3d array containing an image, return the blue channel'''
    blue = []
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            blue.append(img[i][j][2])
    return np.array(blue).reshape(img.shape[0],img.shape[1])

def GetRedPixels(img):
    '''given a numpy 3d array containing an image, return the red channel'''
    red = []
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            red.append(img[i][j][0])
    return np.array(red).reshape(img.shape[0],img.shape[1])

def ColorToGreyscale(img):
    '''given a numpy 3d array containing an image, return the same image in greyscale'''
    red = np.array(GetRedPixels(img),dtype=np.float64)
    green = np.array(GetGreenPixels(img),dtype=np.float64)
    blue = np.array(GetBluePixels(img),dtype=np.float64)
    averageVals = (red + green + blue) / 3
    out = normalize(averageVals)
    return out

def normalize(array):
    '''given a 2d array, return the normalized uint8 array'''
    normalized = []
    maxVal = np.amax(array)
    minVal = np.amin(array)
    for currRow in range(0, len(array)):
        currCol = array[currRow]
        for val in currCol:
            normVal = 255*((val-minVal)/(maxVal-minVal))  # normalize each value, then scale by 255
            normalized.append(normVal)
    normalizedArray = np.array(normalized, dtype=np.uint8)
    return normalizedArray.reshape(array.shape[0], array.shape[1])

def GreyscaleToMonochrome(greyscale):
    '''given a greyscale image, return a monochrome image by thresholding'''
    outImg = []                                     # new array to not overwrite original image
    for currRow in range(0, len(greyscale)):
        newRow = []
        for currCol in range(0, len(greyscale[0])):
            if greyscale[currRow][currCol] < 128:
                newRow.append(0)
            else:
                newRow.append(255)
        outImg.append(newRow)
    return np.array(outImg)

def histogram(greyscale):
    '''given a grayscale image, return a histogram for all of its values'''
    h = {}                                      # initialize dictionary for each value
    for currRow in range(0, len(greyscale)):
        for currCol in range(0, len(greyscale[0])):
            val = greyscale[currRow][currCol]
            if val not in h: 
                h[val] = 1
            else:
                h[val] += 1
    return sortedHistogram(h)

def sortedHistogram(histogram):
    out = {}
    h_sorted = sorted(histogram)
    for key in h_sorted:
        out[key] = histogram[key] 
    return out

def CumulativeDistributionFunction(histogram):
    out = {}
    total = sum(histogram.values())
    running = 0
    for key in histogram:
        currProb = histogram[key] #/total
        out[key] = running + currProb
        running += currProb
    return out

def hist_norm(greyscale):
    normalized = {}
    hist = histogram(greyscale)
    cdf = CumulativeDistributionFunction(hist)
    cdf_min = min(cdf.values())
    M=greyscale.shape[0]
    N=greyscale.shape[1]
    for key in cdf:
        normalized[key] = round(255*(cdf[key] - cdf_min) / ((M*N) - cdf_min))
    return normalized

def rescale(greyscale):
    h_norm = hist_norm(greyscale)
    '''given a greyscale image, return a monochrome image by thresholding'''
    outImg = []                                     # new array to not overwrite original image
    for currRow in range(0, len(greyscale)):
        newRow = []
        for currCol in range(0, len(greyscale[0])):
            val = greyscale[currRow][currCol]
            newRow.append(h_norm[val])
        outImg.append(newRow)
    return np.array(outImg)

if __name__ == "__main__":
    img = loadppm("../images/simple.ascii.ppm")
    greyscale = ColorToGreyscale(img)
    m = hist_norm(greyscale)
    print(greyscale)
    print(rescale(greyscale))

