import os
import sys
import cv2 as cv
import math
import numpy as np
import matplotlib.pyplot as plt
from edge import computeFeatureResponse, genAutoCorrelationMatrix, displayRangeLCS

#Generate histogram of hues in image
#Input: BGR image and number of bins for the histogram
#Output: Array of frequency of hue values in image
def imgHistogram(img, nBins=256):
    #Declare array to hold histogram
    histHSV = np.zeros(nBins, dtype=np.uint64)
    
    #Extract image size attributes
    imgHeight, imgWidth = img.shape[:2]
    
    #Determine bin size for hue range
    binWidth = 256 / nBins
    
    #Iterate through image pixels and increment the appropriate bin
    for y in range(imgHeight):
        for x in range(imgWidth):
            pixelBin = int(img[y][x] / binWidth)
            histHSV[pixelBin] = histHSV[pixelBin] + 1
            
    #Return histogram
    return histHSV

#Function to get mean of histogram values
#Input: Histogram, pixelCount, boolean indicating whether hi or lo image
#Output: Mean value
def histMean(imgHist, pixelCount, nBins=256):
    #Initialize mean
    mean = 0
    
    #Alter range variables to exclude dead bin
    nBins -= 1
    
    #Iterate through histogram and add weighted contribution of pixels in each bin
    for x in range(nBins):
        mean += x * (imgHist[x] / pixelCount)
        
    #Return mean
    return mean

#Wrapper function for recursive Otsu on HSV
#Input: Image
#Output:
def otsuWrapper(inputImg):
    #Get size attributes of image
    imgHeight, imgWidth = inputImg.shape[:2]
    
    #Generate histogram of image
    imgHist = imgHistogram(inputImg)
    
    #Get pixel count, altered for dead bins
    pixelCount = imgHeight * imgWidth
    pixelCount -= imgHist[255]
    
    #Calculate image mean
    imgMean = histMean(imgHist, pixelCount)
    
    #Calculate starting values for qOne and variation squared
    qOne = imgHist[0] / pixelCount
        
    varSquare = qOne * (1 - qOne) * pow(0 - 1, 2)
    
    #Call recursive function
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
def gray2HiLo(inputImg, imgGray, thresh): 
    #Extract image size information
    imgHeight, imgWidth = inputImg.shape[:2]
    
    #Create images to hold pixels above and below threshold
    imgHi = np.zeros((imgHeight, imgWidth, 3), dtype=np.uint8)
    imgLo = np.zeros((imgHeight, imgWidth, 3), dtype=np.uint8)
    
    #Declare variables to count pixels
    countLo = 0
    countHi = 0
    
    #Iterate through image and modify each pixel according to threshold
    for y in range(imgHeight):
        for x in range(imgWidth):
            if (imgGray[y][x] > thresh):
                countHi += 1
                imgHi[y][x] = inputImg[y][x]
                imgLo[y][x] = [255, 255, 255]
            else:
                countLo += 1
                imgHi[y][x] = [255, 255, 255]
                imgLo[y][x] = inputImg[y][x]
                
    #Return binary image
    return imgHi, countHi, imgLo, countLo

#Function to count number of useful edge pixels in image
def countEdges(img, rVals, edgeThresh=50000):
    #Extract image size information
    imgHeight, imgWidth = img.shape[:2]
    
    #Declare variable to hold edge count
    edgeCount = 0
    
    #Iterate through image and increment edge count if non-dead pixel gives strong response
    for y in range(imgHeight):
        for x in range(imgWidth):
            if (img[y][x][0] != 255 and rVals[y][x] > edgeThresh):
                edgeCount += 1
                
    #Return edge count
    return edgeCount

#Function to run Otsu until threshold value converges
def multiOtsu(img, previousThresh):
    #Convert image to gray
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #Apply Otsu to image
    thresh = otsuWrapper(imgGray)
    
    #Get hi and lo thresholded images
    imgHi, pixelCountHi, imgLo, pixelCountLo = gray2HiLo(img, imgGray, thresh)
    
    plt.imsave(outPath + 'Hi' + str(thresh) + '.png', imgHi)
    plt.imsave(outPath + 'Lo' + str(thresh) + '.png', imgLo)
    
    #Generate auto-correlation matrices
    matrixHi = genAutoCorrelationMatrix(imgHi, 2.5)
    matrixLo = genAutoCorrelationMatrix(imgLo, 2.5)

    #Calculate feature response and save to output
    rValsHi = computeFeatureResponse(matrixHi)
    plt.imsave(outPath + 'EdgeHi' + str(thresh) + '.png', displayRangeLCS(rValsHi))
    rValsLo = computeFeatureResponse(matrixLo)
    plt.imsave(outPath + 'EdgeLo' + str(thresh) + '.png', displayRangeLCS(rValsLo))
    
    print(f'Threshold: {thresh}')
    
    countHi = countEdges(imgHi, rValsHi, 50000)
    countLo = countEdges(imgLo, rValsLo, 50000)
    print(f'Hi: {countHi} Pixel: {pixelCountHi}')
    print(f'Lo: {countLo} Pixel: {pixelCountLo}')
    
    #Choose image for next iteration
    if (pixelCountHi == 0):
        imgFinal = imgLo
    elif (pixelCountLo == 0):
        imgFinal = imgHi
    elif (countHi / pixelCountHi > countLo / pixelCountLo):
        imgFinal = imgHi
        print("hi!!")
    else:
        imgFinal = imgLo
        print("lo!!")
    
    #Check if thresh value is the same as previous
    if (thresh == previousThresh):
        return imgFinal
    else:
        return multiOtsu(imgFinal, thresh)
    
#Function to remove edges of extracted image
def cropEdges(img):
    #Extract image size attributes
    imgHeight, imgWidth = img.shape[:2]
    
    #Ignore 10% of pixels on each side
    borderX = int(imgWidth / 20)
    borderY = int(imgHeight / 20)
    return img[borderY:imgHeight - borderY, borderX:imgWidth - borderX]

#Apply pre-processing pipeline to image
def preprocessBook(inputImg, outPath):
    #Make directory to save output images
    if (not os.path.exists(outPath)):
        os.mkdir(outPath)
        
    #Crop image edges
    imgCrop = cropEdges(inputImg)
    
    #Apply multiple iterations of Otsu
    imgText = multiOtsu(imgCrop, 0)
    
    plt.imsave(outPath + 'final.png', imgText)

'''#Get paths for input and output files
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
outPath = str(srcDir + '\\' + sys.argv[2])

#Import input image
inputImg = plt.imread(inPath)

#Apply pre-processing to input image, then save to outPath
preprocessBook(inputImg, outPath)'''