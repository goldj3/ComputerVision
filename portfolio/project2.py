import numpy as np
import cv2
from utils import vis_hybrid_image, load_image, save_image


def my_imfilter(image, fil):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that I can finish grading 
   before the heat death of the universe. 
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

  assert fil.shape[0] % 2 == 1
  assert fil.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###

  img = np.array(image)
  rowsToPad= int((fil.shape[0] - 1)/2)          
  colsToPad = int((fil.shape[1] - 1)/2)
  red = GetRedPixels(img)      # separates each channel from original image 
  blue = GetBluePixels(img)
  green = GetGreenPixels(img)
  red_padded = np.pad(red, ((rowsToPad,rowsToPad),(colsToPad,colsToPad)), 'edge')     # pad each channel
  green_padded = np.pad(green, ((rowsToPad,rowsToPad),(colsToPad,colsToPad)), 'edge')
  blue_padded = np.pad(blue, ((rowsToPad,rowsToPad),(colsToPad,colsToPad)), 'edge')
  convolvedRed = convolve(red,red_padded,fil)  # convolve each padded channel
  convolvedGreen = convolve(green,green_padded,fil)
  convolvedBlue = convolve(blue, blue_padded, fil)
  out_img = np.dstack((convolvedRed,convolvedGreen,convolvedBlue))     # brings all the convolved color channels together
  return out_img

  ### END OF STUDENT CODE ####
  ############################

  #return filtered_image
  

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  ############################
  ### TODO: YOUR CODE HERE ###
  img1_lowFreq = my_imfilter(image1, filter)
  img2_lowFreq = my_imfilter(image2, filter)
  low_frequencies = np.clip(img1_lowFreq, 0, 1)
  high_frequencies = np.subtract(image2, img2_lowFreq)
  high_frequencies = np.clip(high_frequencies, 0, 1)
  hybrid_image = np.add(low_frequencies, high_frequencies)
  ### END OF STUDENT CODE ####
  ############################

  return low_frequencies, high_frequencies, hybrid_image


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


def convolve(orig_channel, padded_channel, fil):
  '''convolves a channel with a filter, given the original channel, the padded channel, and the filter'''
  out = []
  padded_rows = fil.shape[0]
  padded_cols = fil.shape[1]
  for i in range(0, orig_channel.shape[0]):                             # goes through # of rows of original image, does not convolve padded values
    for j in range(0, orig_channel.shape[1]):                           # goes through # of columns of original image, does not convolve padded values
        curMatrix = padded_channel[i:i+padded_rows:, j:j+padded_cols:]     # gets a matrix from the padded channel that is the same size of the filter
        out.append(np.sum(np.multiply(curMatrix,fil)))                     # multiplies the same positions of the filter and the original channel, sums all the multiplied values to get new pixel value for the proper position
  return np.array(out).reshape((orig_channel.shape[0], orig_channel.shape[1]))  # all padded vals are gone, putting convolved values for each pixel in original place

  

if __name__ == "__main__":
  image1 = load_image('../project2/images/einstein.bmp')
  image2 = load_image('../project2/images/marilyn.bmp')
  cutoff_frequency = 7
  filter = cv2.getGaussianKernel(ksize=cutoff_frequency*4+1,
                               sigma=cutoff_frequency)
  filter = np.dot(filter, filter.T)
  low_frequencies, high_frequencies, hybrid_image = create_hybrid_image(image1, image2, filter)
  print(low_frequencies)
    