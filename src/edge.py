import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Function to compute x-gradient for image
#Input: Single-channel image
#Output: x-gradient image (POSSIBLE MIN: = -1020 POSSIBLE MAX = 1020)
def xGrad(img):
    #Calculate image filtered with y-direction Sobel
    return cv.Sobel(img, cv.CV_64F, 1, 0, 1)

#Function to compute y-gradient for image
#Input: Single-channel image
#Output: y-gradient image (POSSIBLE MIN: = -1020 POSSIBLE MAX = 1020)
def yGrad(img):
    #Calculate image filtered with y-direction Sobel
    return cv.Sobel(img, cv.CV_64F, 0, 1, 1)

#Function to compute top-left tensor element
#Input: RGB image and sigma value for Gaussian filter
#Output: Top-left tensor element
def topLeftElem(img, sigma):
    #Compute first x-derivative of image for each color channel
    xGradB = xGrad(img[:,:,0])
    xGradG = xGrad(img[:,:,1])
    xGradR = xGrad(img[:,:,2])
    
    #Raise all array elements to power of two for each channel (POSSIBLE MIN: = 0 POSSIBLE MAX = 1040400)
    xGradSquareB = np.power(xGradB, 2)
    xGradSquareG = np.power(xGradG, 2)
    xGradSquareR = np.power(xGradR, 2)
    
    #Sum values
    xGradSum = np.add(xGradSquareB, np.add(xGradSquareG, xGradSquareR))
    
    #Sum each color channel to produce final tensor and return (POSSIBLE MIN: = 0 POSSIBLE MAX = 3121200)
    return cv.GaussianBlur(xGradSum, (0,0), sigma)

#Function to compute gradient magnitude
#Input: x- and y-gradients
#Output: Gradient magnitude image
def gradMag(xGrad, yGrad):
    return np.sqrt(np.pow(xGrad, 2) + np.pow(yGrad, 2))

#Function to compute bottom-right tensor element
#Input: RGB image and sigma value for Gaussian filter
#Output: Bottom-right tensor element
def bottomRightElem(img, sigma):
    #Compute first y-derivative of image for each color channel
    yGradB = yGrad(img[:,:,0])
    yGradG = yGrad(img[:,:,1])
    yGradR = yGrad(img[:,:,2])
    
    #Raise all array elements to power of two for each channel (POSSIBLE MIN: = 0 POSSIBLE MAX = 1040400)
    yGradSquareB = np.power(yGradB, 2)
    yGradSquareG = np.power(yGradG, 2)
    yGradSquareR = np.power(yGradR, 2)
    
    #Sum values
    yGradSum = np.add(yGradSquareB, np.add(yGradSquareG, yGradSquareR))
    
    #Sum each color channel to produce final tensor and return (POSSIBLE MIN: = 0 POSSIBLE MAX = 3121200)
    return cv.GaussianBlur(yGradSum, (0,0), sigma)

#Function to compute top-left/bottom-right tensor elements
#Input: RGB image and sigma value for Gaussian filter
#Output: Top-left/botton-right tensor element
def otherElems(img, sigma):
    #Compute first x-derivative of image for each color channel
    xGradB = xGrad(img[:,:,0])
    xGradG = xGrad(img[:,:,1])
    xGradR = xGrad(img[:,:,2])
    
    #Compute first y-derivative of image for each color channel
    yGradB = yGrad(img[:,:,0])
    yGradG = yGrad(img[:,:,1])
    yGradR = yGrad(img[:,:,2])
    
    #Multiply x- and y-derivatives for each channel (POSSIBLE MIN: = -1040400 POSSIBLE MAX = 1040400)
    xyMultB = np.multiply(xGradB, yGradB)
    xyMultG = np.multiply(xGradG, yGradG)
    xyMultR = np.multiply(xGradR, yGradR)
    
    #Sum values
    xyMultSum = np.add(xyMultB, np.add(xyMultG, xyMultR))
    
    #Sum each color channel to produce final tensor and return (POSSIBLE MIN: = -3121200 POSSIBLE MAX = 3121200)
    return cv.GaussianBlur(xyMultSum, (0,0), sigma)

#Function to calculate determinant of tensor matrix
#Input: Auto-correlation matrix
#Output: Determinant
def tensorDeterminant(matrix):
    return np.subtract(np.multiply(matrix[0], matrix[3]), np.multiply(matrix[1], matrix[2])) 

#Function to calculate trace of tensor matrix
#Input: Top-left and bottom-right tensor elements
#Output: Trace (POSSIBLE MIN: = 0 POSSIBLE MAX = 6242400)
def tensorTrace(topLeft, bottomRight):
    return np.add(topLeft, bottomRight)

#Function to construct autocorrelation matrix from image
#Input: Source image and sigma value for Gaussian smoothing
#Output: Array of images [top-left, top-right, bottom-left, bottom-right]
def genAutoCorrelationMatrix(img, sigma):
    #Initiate output array
    matrix = []
    
    #Add each matrix element in order
    matrix.append(topLeftElem(img, sigma))
    matrix.append(otherElems(img, sigma))
    matrix.append(otherElems(img, sigma))
    matrix.append(bottomRightElem(img, sigma))
    
    #Return autocorrelation matrix
    return matrix

#Function to detect corners using Harris-Stephens Detector
#Input: Auto-correlation matrix of input image
#Output: Array of R values for whole image
def computeFeatureResponse(matrix):
    #Set value of empirical constant [0.04, 0.06]
    alpha = 0.05
    
    return np.subtract(tensorDeterminant(matrix), np.multiply(alpha, np.power(tensorTrace(matrix[0], matrix[3]), 2)))

#Function to perform max LCS to display range
#Input: Input image
#Output: Displayable image
def displayRangeLCS(img):
    if (np.min(img) == np.max(img)):
        return img
    return (img - np.min(img)) * (255 / (np.max(img) - np.min(img)))