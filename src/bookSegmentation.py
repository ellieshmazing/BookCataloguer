import os
import sys
import time
import cv2 as cv
import pickle as pkl
import matplotlib.pyplot as plt
from bookSeeding import findTextRegions, cullDetections, drawDetectionsCropped, isolateShelf
from bookSuperpixelization import visualizeSuperpixels, groupSuperpixels
    
#Extract book spines
def extractBooks(inputImg, outPath, sigma=2, regionSize=1000):
    #Make directory to save output images
    if (not os.path.exists(outPath)):
        os.mkdir(outPath)
        
    #Detect text in input image
    print("Detecting text regions...")
    ocrResults = findTextRegions(inputImg, outPath)
    
    #Cull bad detections
    print("Culling bad/repeat detections...")
    ocrResults = cullDetections(ocrResults)
    plt.imsave(outPath + 'Culled.jpg', drawDetectionsCropped(inputImg, ocrResults, 0, 0))
    
    #Extract shelf
    print("Isolating bookshelf...")
    imgShelf, xDiff, yDiff = isolateShelf(inputImg, ocrResults)
    plt.imsave(outPath + 'Shelf.jpg', imgShelf)
    
    #Convert image to LAB for SLIC effectiveness
    imgLab = cv.cvtColor(imgShelf, cv.COLOR_BGR2Lab)
    
    #Apply Gaussian blur to image
    imgLab = cv.GaussianBlur(imgLab, (3,3), sigma)
    plt.imsave(outPath + 'Lab.jpg', imgLab)
    
    #Generate Superpixel object from image
    print("Generating Superpixel image...")
    objSP = cv.ximgproc.createSuperpixelSLIC(imgLab, cv.ximgproc.MSLIC, regionSize)
    objSP.iterate(1)
    
    #Generate Superpixel boundary mask
    spBoundaries = objSP.getLabelContourMask()
    cv.imwrite(outPath + 'SPBound.jpg', spBoundaries)
    
    #Get Superpixel labels and count
    labelSP = objSP.getLabels()
    numSP = objSP.getNumberOfSuperpixels()
    
    #Generate Superpixel image and determine dominant color for each
    print("Generating Superpixel visual...")
    imgSP, labelColors = visualizeSuperpixels(imgLab, labelSP, numSP)
    plt.imsave(outPath + 'SP.jpg', imgSP)
    plt.imsave(outPath + 'SPBox.jpg', drawDetectionsCropped(imgSP, ocrResults, xDiff, yDiff))
    
    #Group Superpixels according to the book they belong to
    print("Extracting book spines...")
    bookImgs = groupSuperpixels(imgShelf, outPath, ocrResults, xDiff, yDiff, spBoundaries, numSP, labelSP, labelColors)
    
    #Save book images
    with open(outPath + "bookImages.pickle", "wb") as f:
        pkl.dump(bookImgs, f)
    
#Get paths for input and output files
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
outPath = str(srcDir + '\\' + sys.argv[2])

#Import input image
inputImg = plt.imread(inPath)

startTime = time.time()

#Extract all book spines in image
extractBooks(inputImg, outPath)

endTime = time.time()
print(f'Execution time: {endTime - startTime}')