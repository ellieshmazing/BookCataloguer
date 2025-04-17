import os
import sys
import cv2 as cv
import math
import numpy as np
import matplotlib.pyplot as plt

#Generate histogram of hues in image
#Input: BGR image and number of bins for the histogram
#Output: Array of frequency of hue values in image
def hsvHistogram(imgBGR, nBins=256):
    #Convert image to HSV
    imgHSV = cv.cvtColor(imgBGR, cv.COLOR_BGR2HSV)
    
    #Declare array to hold histogram
    histHSV = np.zeros(nBins, dtype=np.uint64)
    
    #Extract image size attributes
    imgHeight, imgWidth = imgHSV.shape[:2]
    
    #Determine bin size for hue range
    binWidth = 256 / nBins
    
    #Iterate through image pixels and increment the appropriate bin
    for y in range(imgHeight):
        for x in range(imgWidth):
            pixelBin = int(imgHSV[y][x][0] / binWidth)
            histHSV[pixelBin] = histHSV[pixelBin] + 1
            
    #Return histogram
    return histHSV

#Function to get mean of histogram values
#Input: Histogram, pixelCount, boolean indicating whether hi or lo image
#Output: Mean value
def histMean(imgHist, pixelCount, hiBool, nBins=256):
    #Initialize mean
    mean = 0
    
    #Alter range variables depending on hiBool (exclude 0 if hi, 255 if low)
    start = 0
    if (hiBool):
        start += 1
    else:
        nBins -= 1
    
    #Iterate through histogram and add weighted contribution of pixels in each bin
    for x in range(start, nBins):
        mean += x * (imgHist[x] / pixelCount)
        
    #Return mean
    return mean

#Wrapper function for recursive Otsu on HSV
#Input: Image
#Output:
def otsuWrapper(inputImg, hiBool):
    #Get size attributes of image
    imgHeight, imgWidth = inputImg.shape[:2]
    
    #Generate histogram of image
    imgHist = hsvHistogram(inputImg)
    
    #Get pixel count, altered for dead bins
    pixelCount = imgHeight * imgWidth
    if (hiBool):
        pixelCount -= imgHist[0]
    else:
        pixelCount -= imgHist[255]
    
    #Calculate image mean
    imgMean = histMean(imgHist, pixelCount, hiBool)
    
    #Calculate starting values for qOne and variation squared
    if (hiBool):
        qOne = imgHist[1] / pixelCount
    else:
        qOne = imgHist[0] / pixelCount
        
    varSquare = qOne * (1 - qOne) * pow(0 - 1, 2)
    
    #Call recursive function
    if (hiBool):
        return recursiveOtsu(imgHist, pixelCount, imgMean, qOne, varSquare, thresh=2)
    else:
        return recursiveOtsu(imgHist, pixelCount, imgMean, qOne, varSquare)
    
#Recursive Otsu function
#Input: Histogram of image values, count of pixels, mean pixel value
#Output: Threshold value for binary mask
def recursiveOtsu(imgHist, pixelCount, mean, qOne, varSquare, thresh=1, uOne=0):
    #Calculate next values for pOne and qOne
    pOnePlus = imgHist[thresh] / pixelCount
    qOnePlus = qOne + pOnePlus
    
    #Calculate the left class mean (with default to 0 if qOnePlus is 0 to avoid divide by zero error)
    if (qOnePlus > 0):
        uOnePlus = ((qOne * uOne) + (thresh * pOnePlus)) / qOnePlus
    else:
        uOnePlus = 0
    
    #Calculate the right class mean
    uTwoPlus = (mean - (qOnePlus * uOnePlus)) / (1 - qOnePlus)
    
    #Calculate the between-class variance with current threshold
    varSquarePlus = qOnePlus * (1 - qOnePlus) * pow(uOnePlus - uTwoPlus, 2)
        
    #If variance decreases, return previous value as it was maximum
    #This is due to the usage of binarization, as the variance will always be an inverted polynomial. With more classes,
    #ending execution early would not be possible
    if (varSquarePlus < varSquare):
        return thresh - 1
    
    #End execution if threshold is maximum value
    if (thresh == 255):
        return thresh
        
    #Execute for next threshold value
    return recursiveOtsu(imgHist, pixelCount, mean, qOnePlus, varSquarePlus, thresh+1, uOnePlus)

#Apply threshold to image
def hsv2binaryHiLo(imgHSV, thresh): 
    #Extract image size information
    imgHeight, imgWidth = imgHSV.shape[:2]
    
    #Initialize variables to hold pixel counts
    countHi = 0
    countLo = 0
    
    #Create images to hold pixels above and below threshold
    imgHSVHi = np.zeros((imgHeight, imgWidth, 3), dtype=np.uint8)
    imgHSVLo = np.zeros((imgHeight, imgWidth, 3), dtype=np.uint8)
    
    #Iterate through image and modify each pixel according to threshold
    for y in range(imgHeight):
        for x in range(imgWidth):
            if (imgHSV[y][x][0] > thresh):
                countHi += 1
                imgHSVHi[y][x] = imgHSV[y][x]
                imgHSVLo[y][x] = [255, 255, 255]
            else:
                countLo += 1
                imgHSVHi[y][x] = [0, 0, 0]
                imgHSVLo[y][x] = imgHSV[y][x]
                
    #Return binary image
    return imgHSVHi, countHi, imgHSVLo, countLo

#Function to run Otsu until threshold value converges
def multiOtsu(inputImgHSV, previousThresh, hiBool):
    #Apply Otsu to image
    thresh = otsuWrapper(inputImgHSV, hiBool)
    
    #Get hi and lo thresholded images
    imgHSVHi, countHi, imgHSVLo, countLo = hsv2binaryHiLo(inputImgHSV, thresh)
    
    plt.imsave(outPath + 'Hi' + str(thresh) + '.png', imgHSVHi)
    plt.imsave(outPath + 'Lo' + str(thresh) + '.png', imgHSVLo)
    
    #Choose image for next iteration
    if (countHi > countLo):
        imgFinal = imgHSVHi
        hiBool = True
    else:
        imgFinal = imgHSVLo
        hiBool = False
    
    #Check if thresh value is the same as previous
    if (thresh == previousThresh):
        return imgFinal
    else:
        return multiOtsu(imgFinal, thresh, hiBool)

#Apply pre-processing pipeline to image
def preprocessBook(inputImg, outPath):
    #Make directory to save output images
    if (not os.path.exists(outPath)):
        os.mkdir(outPath)
        
    #Convert image to HSV color space
    imgHSV = cv.cvtColor(inputImg, cv.COLOR_BGR2HSV)
    
    #Apply multiple iterations of Otsu
    imgText = multiOtsu(imgHSV, 0, True)
    
    plt.imsave(outPath + 'final.png', imgText)

#Get paths for input and output files
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
outPath = str(srcDir + '\\' + sys.argv[2])

#Import input image
inputImg = plt.imread(inPath)

#Apply pre-processing to input image, then save to outPath
preprocessBook(inputImg, outPath)