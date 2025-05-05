import os
import sys
import cv2 as cv
import math
import easyocr
import numpy as np
import pickle as pkl
from PIL import Image
import matplotlib.pyplot as plt
from preprocessing import otsuWrapper

#Draw OCR detections on image
def drawDetections(img, ocrResults):
    #Copy image to annotate
    annotatedImg = img.copy()
    
    #Iterate through results and draw box for each
    for bbox in ocrResults:
        pts = np.array(bbox, dtype=np.int32)
        cv.polylines(annotatedImg, [pts], isClosed=True, color=(0,255,0))
        
    #Return annotated image
    return annotatedImg

#Draw OCR detections after crop
def drawDetectionsCropped(img, ocrResults, xDiff, yDiff):
    #Copy image to annotate
    annotatedImg = img.copy()
    
    #Iterate through results and draw box for each
    for bbox in ocrResults:
        pts = np.array(bbox, dtype=np.int32)
        
        #Move bbox by amount of image removed
        for i in range(4):
            pts[i][0] -= xDiff;
            pts[i][1] -= yDiff;
            
        #Draw boundary
        cv.polylines(annotatedImg, [pts], isClosed=True, color=(0,255,0))
        
    #Return annotated image
    return annotatedImg

#Apply OCR to input image
def findTextRegions(inputImg):
    #Initialize OCR reader
    reader = easyocr.Reader(['en'], gpu=False)
    
    #Rotate image for all orientations
    input90 = np.rot90(inputImg, axes=(0,1))
    input180 = np.rot90(input90, axes=(0,1))
    input270 = np.rot90(input180, axes=(0,1))
    
    #Declare array for all results, where index correlates to 90 degrees of rotation
    ocrResults = []
    
    #Run reader on all image orientations
    ocrResults.append(reader.readtext(inputImg))
    ocrResults.append(reader.readtext(input90))
    ocrResults.append(reader.readtext(input180))
    ocrResults.append(reader.readtext(input270))
    
    #Save ocrResults
    with open(outPath + "ocrResults.pickle", "wb") as f:
        pkl.dump(ocrResults, f)
    
    #Reorient results to same coordinate system
    ocrResults = reorientResults(ocrResults, inputImg)
    
    #Save detections for each orientation
    plt.imsave(outPath + '0.jpg', drawDetections(inputImg, ocrResults[0]))
    plt.imsave(outPath + '90.jpg', drawDetections(inputImg, ocrResults[1]))
    plt.imsave(outPath + '180.jpg', drawDetections(inputImg, ocrResults[2]))
    plt.imsave(outPath + '270.jpg', drawDetections(inputImg, ocrResults[3]))
    
    #Combine detections from all orientations into single image
    annotated = drawDetections(inputImg, ocrResults[0])
    annotated = drawDetections(annotated, ocrResults[1])
    annotated = drawDetections(annotated, ocrResults[2])
    plt.imsave(outPath + 'all.jpg',drawDetections(annotated, ocrResults[3]))
    
    #Return detection results
    return ocrResults

#Convert OCR results to original orientation and remove low confidence entries
def reorientResults(ocrResults, inputImg, confThresh=0.3):
    #Extract image size attributes
    imgHeight, imgWidth = inputImg.shape[:2]
    
    #Declare array to hold reoriented results
    ocrResultsNew = [[],[],[],[]]
    
    #Remove low confidence results from original
    for (bbox, _, conf) in ocrResults[0]:
        if (conf > confThresh):
            ocrResultsNew[0].append([bbox, conf])
            
    #Remove low confidence results and reorient remaining from 90 degree
    for (bbox, _, conf) in ocrResults[1]:
        if (conf > confThresh):
            #Declare array to hold new bbox coordinates
            bboxNew = [[0,0],[0,0],[0,0],[0,0]]
            
            #Translate x coordinates
            bboxNew[0][0] = imgWidth - bbox[0][1]
            bboxNew[1][0] = imgWidth - bbox[1][1]
            bboxNew[2][0] = imgWidth - bbox[2][1]
            bboxNew[3][0] = imgWidth - bbox[3][1]
            
            #Translate y coordinates
            bboxNew[0][1] = bbox[0][0]
            bboxNew[1][1] = bbox[1][0]
            bboxNew[2][1] = bbox[2][0]
            bboxNew[3][1] = bbox[3][0]
            
            #Append to coordinate list
            ocrResultsNew[1].append([bboxNew, conf])
            
    #Remove low confidence results and reorient remaining from 180 degree
    for (bbox, _, conf) in ocrResults[2]:
        if (conf > confThresh):
            #Declare array to hold new bbox coordinates
            bboxNew = [[0,0],[0,0],[0,0],[0,0]]
            
            #Translate x coordinates
            bboxNew[0][0] = imgWidth - bbox[0][0]
            bboxNew[1][0] = imgWidth - bbox[1][0]
            bboxNew[2][0] = imgWidth - bbox[2][0]
            bboxNew[3][0] = imgWidth - bbox[3][0]
            
            #Translate y coordinates
            bboxNew[0][1] = imgHeight - bbox[0][1]
            bboxNew[1][1] = imgHeight - bbox[1][1]
            bboxNew[2][1] = imgHeight - bbox[2][1]
            bboxNew[3][1] = imgHeight - bbox[3][1]
            
            #Append to coordinate list
            ocrResultsNew[2].append([bboxNew, conf])
            
    #Remove low confidence results and reorient remaining from 270 degree
    for (bbox, _, conf) in ocrResults[3]:
        if (conf > confThresh):
            #Declare array to hold new bbox coordinates
            bboxNew = [[0,0],[0,0],[0,0],[0,0]]
            
            #Translate x coordinates
            bboxNew[0][0] = bbox[0][1]
            bboxNew[1][0] = bbox[1][1]
            bboxNew[2][0] = bbox[2][1]
            bboxNew[3][0] = bbox[3][1]
            
            #Translate y coordinates
            bboxNew[0][1] = imgHeight - bbox[0][0]
            bboxNew[1][1] = imgHeight - bbox[1][0]
            bboxNew[2][1] = imgHeight - bbox[2][0]
            bboxNew[3][1] = imgHeight - bbox[3][0]
            
            #Append to coordinate list
            ocrResultsNew[3].append([bboxNew, conf])
            
    #Return reoriented ocrResults
    return ocrResultsNew
            

#Find corner coordinates for ocrResults
def findCorners(inputImg, ocrResults):
    #Extract image size attributes
    imgHeight, imgWidth = inputImg.shape[:2]
    
    #Declare vars to hold highest and lowest x and y
    xLow = np.inf
    xHigh = 0
    yLow = np.inf
    yHigh = 0
    
    #Iterate through array to find lowest and highest x and y
    for bbox in ocrResults:
        for coord in bbox:
            xLow = np.min((xLow, coord[0]))
            xHigh = np.max((xHigh, coord[0]))
                    
            yLow = np.min((yLow, coord[1]))
            yHigh = np.max((yHigh, coord[1]))
                    
    #Return full bounding box
    return [int(np.max((xLow, 0))), int(np.min((xHigh, imgWidth))), int(np.max((yLow, 0))), int(np.min((yHigh, imgHeight)))]

#Isolate shelf section of image
def isolateShelf(inputImg, ocrResults):
    #Find corner coordinates for each iteration
    coordCorner = findCorners(inputImg, ocrResults)
    
    #Extract shelf bounding box
    imgShelf = inputImg[coordCorner[2]:coordCorner[3], coordCorner[0]:coordCorner[1], :]
    
    #Calculate xDiff and yDiff (for converting from original coordinate space to cropped image)
    xDiff = coordCorner[0]
    yDiff = coordCorner[2]
    
    #Return shelf image, xDiff, and yDiff
    return imgShelf, xDiff, yDiff

#Check if two boxes overlap
def checkOverlap(bbox1, bbox2):
    #Convert to Numpy arrays to allow slicing
    bbox1 = np.array(bbox1)
    bbox2 = np.array(bbox2)
    
    #Check if bbox1 coords are within bbox2
    for coord in bbox1:
        if(coord[0] > np.min(bbox2[:,0]) and coord[0] < np.max(bbox2[:,0]) and coord[1] > np.min(bbox2[:,1]) and coord[1] < np.max(bbox2[:,1])):
            return True
        
    #Check if bbox2 coords are within bbox1
    for coord in bbox2:
        if (coord[0] > np.min(bbox1[:,0]) and coord[0] < np.max(bbox1[:,0]) and coord[1] > np.min(bbox1[:,1]) and coord[1] < np.max(bbox1[:,1])):
            return True

    #Return False otherwise
    return False    
    
#Cull overlapping detections
def cullDetections(ocrResults):
    #Initialize array to hold culled results
    ocrResultsNew = []
    
    #Add non-overlapping coordinates from original image orientation
    for (bbox1, conf1) in ocrResults[0]:
        #Initialize boolean indicating passage
        bboxUnique = True
        
        for (bbox2, conf2) in ocrResults[0]:
            if (checkOverlap(bbox1, bbox2) and conf1 < conf2):
                bboxUnique = False
        
        #Check against 90 degree
        for (bbox2, conf2) in ocrResults[1]:
            if (checkOverlap(bbox1, bbox2) and conf1 < conf2):
                bboxUnique = False
                
        for (bbox2, conf2) in ocrResults[2]:
            if (checkOverlap(bbox1, bbox2) and conf1 < conf2):
                bboxUnique = False
                
        for (bbox2, conf2) in ocrResults[3]:
            if (checkOverlap(bbox1, bbox2) and conf1 < conf2):
                bboxUnique = False
                
        if (bboxUnique):
            ocrResultsNew.append(bbox1)
            
    #Add non-overlapping coordinates from 90 degree image orientation
    for (bbox1, conf1) in ocrResults[1]:
        #Initialize boolean indicating passage
        bboxUnique = True
        
        #Check against 90 degree
        for (bbox2, conf2) in ocrResults[0]:
            if (checkOverlap(bbox1, bbox2) and conf1 < conf2):
                bboxUnique = False
                
        for (bbox2, conf2) in ocrResults[1]:
            if (checkOverlap(bbox1, bbox2) and conf1 < conf2):
                bboxUnique = False
                
        for (bbox2, conf2) in ocrResults[2]:
            if (checkOverlap(bbox1, bbox2) and conf1 < conf2):
                bboxUnique = False
                
        for (bbox2, conf2) in ocrResults[3]:
            if (checkOverlap(bbox1, bbox2) and conf1 < conf2):
                bboxUnique = False
                
        if (bboxUnique):
            ocrResultsNew.append(bbox1)
            
    #Add non-overlapping coordinates from 180 image orientation
    for (bbox1, conf1) in ocrResults[2]:
        #Initialize boolean indicating passage
        bboxUnique = True
        
        #Check against 90 degree
        for (bbox2, conf2) in ocrResults[0]:
            if (checkOverlap(bbox1, bbox2) and conf1 < conf2):
                bboxUnique = False
                
        for (bbox2, conf2) in ocrResults[1]:
            if (checkOverlap(bbox1, bbox2) and conf1 < conf2):
                bboxUnique = False
                
        for (bbox2, conf2) in ocrResults[2]:
            if (checkOverlap(bbox1, bbox2) and conf1 < conf2):
                bboxUnique = False
                
        for (bbox2, conf2) in ocrResults[3]:
            if (checkOverlap(bbox1, bbox2) and conf1 < conf2):
                bboxUnique = False
                
        if (bboxUnique):
            ocrResultsNew.append(bbox1)
            
        #Add non-overlapping coordinates from 270 image orientation
    for (bbox1, conf1) in ocrResults[3]:
        #Initialize boolean indicating passage
        bboxUnique = True
        
        #Check against 90 degree
        for (bbox2, conf2) in ocrResults[0]:
            if (checkOverlap(bbox1, bbox2) and conf1 < conf2):
                bboxUnique = False
                
        for (bbox2, conf2) in ocrResults[1]:
            if (checkOverlap(bbox1, bbox2) and conf1 < conf2):
                bboxUnique = False
                
        for (bbox2, conf2) in ocrResults[2]:
            if (checkOverlap(bbox1, bbox2) and conf1 < conf2):
                bboxUnique = False
                
        for (bbox2, conf2) in ocrResults[3]:
            if (checkOverlap(bbox1, bbox2) and conf1 < conf2):
                bboxUnique = False
                
        if (bboxUnique):
            ocrResultsNew.append(bbox1)
            
    #Return unique bounding boxes
    return ocrResultsNew

#Visualize superpixels
def visualizeSuperpixels(imgShelf, labelSP, numSP):
    #Extract image size attributes
    imgHeight, imgWidth = imgShelf.shape[:2]
    
    #Create array to hold output image
    imgSP = np.zeros((imgHeight, imgWidth, 3), dtype=np.uint8)
    
    #Create array to hold label colors, and set min pos to inf
    labelColors = np.zeros((numSP, 8), dtype=np.float64)
    labelColors[:,4] = np.inf
    labelColors[:,6] = np.inf
    
    #Iterate through image to determine label colors
    for y in range(imgHeight):
        for x in range(imgWidth):
            #Increment SP pixel count
            labelColors[labelSP[y][x]][3] += 1
            
            #Add color to cumulative color labels
            labelColors[labelSP[y][x]][0] += imgShelf[y][x][0]
            labelColors[labelSP[y][x]][1] += imgShelf[y][x][1]
            labelColors[labelSP[y][x]][2] += imgShelf[y][x][2]
            
            #Update position attributes
            labelColors[labelSP[y][x]][4] = np.min((labelColors[labelSP[y][x]][4], x)) #xMin
            labelColors[labelSP[y][x]][5] = np.max((labelColors[labelSP[y][x]][5], x)) #xMax
            labelColors[labelSP[y][x]][6] = np.min((labelColors[labelSP[y][x]][6], y)) #yMin
            labelColors[labelSP[y][x]][7] = np.max((labelColors[labelSP[y][x]][7], y)) #yMax
            
    for i in range(numSP):
        labelColors[i][0] /= labelColors[i][3]
        labelColors[i][1] /= labelColors[i][3]
        labelColors[i][2] /= labelColors[i][3]
    
    for y in range(imgHeight):
        for x in range(imgWidth):
            imgSP[y][x][0] = labelColors[labelSP[y][x]][0]
            imgSP[y][x][1] = labelColors[labelSP[y][x]][1]
            imgSP[y][x][2] = labelColors[labelSP[y][x]][2]
            
    return imgSP, labelColors

#Assign initial superpixels in bounding box
def assignInitialSuperpixels(bbox, xDiff, yDiff, labelSP, thresh=0.1):
    #Determine pixel boundaries of detection box
    yLow = np.round(np.min(bbox[:,1])) - yDiff
    yHigh = np.round(np.max(bbox[:,1])) - yDiff
    xLow = np.round(np.min(bbox[:,0])) - xDiff
    xHigh = np.round(np.max(bbox[:,0])) - xDiff
    
    #Calculate total number of pixels in box
    totalPixels = (yHigh - yLow) * (xHigh - xLow)
    
    #Declare array to hold Superpixel counts
    countSP = {}
    
    #Iterate through all pixels in bbox and increment their SP
    for y in range(int(np.round(yHigh - yLow))):
        for x in range(int(np.round(xHigh - xLow))):
            labelCurr = labelSP[yLow + y][xLow + x]
            
            if labelCurr in countSP:
                countSP[labelCurr] += 1
            else:
                countSP.update({labelCurr: 1})
                
    #Declare array to hold output
    initialSP = []
    
    #If SP above threshold, add label and color to initial array
    for labelCurr, countCurr in countSP.items():
        if ((countCurr / totalPixels) / thresh):
            initialSP.append(labelCurr)
            
    return initialSP

#Get all pixels belonging to book group
def getBookMask(inputImg, bookSP, labelSP):
    #Extract image size attributes
    imgHeight, imgWidth = inputImg.shape[:2]
    
    #Initialize array to hold mask of only book pixels
    bookMask = np.zeros((imgHeight, imgWidth, 3), dtype=np.uint8)
    
    #Iterate through image and copy pixels that belong to the book's Superpixels
    for y in range(imgHeight):
        for x in range(imgWidth):
            if (labelSP[y][x] in bookSP):
                bookMask[y][x] = inputImg[y][x]
                
    #Return extracted bookMask
    return bookMask

#Get book Superpixel list position variables
def getBookSPPos(bookSP, labelColors):
    #Declare array to hold output
    bookSPPos = np.zeros(4)
    bookSPPos[0] = np.inf
    bookSPPos[2] = np.inf
    
    #Check max/min for every SP in bookSP
    for i in range(len(bookSP)):
        bookSPPos[0] = np.min((bookSPPos[0], labelColors[i][4]))
        bookSPPos[1] = np.max((bookSPPos[1], labelColors[i][5]))
        bookSPPos[2] = np.min((bookSPPos[2], labelColors[i][6]))
        bookSPPos[3] = np.max((bookSPPos[3], labelColors[i][7]))
        
    #Return min and max x and ys
    return bookSPPos

#Generate Superpixel adjacency matrix
def generateSuperpixelAdjacencyMatrix(spBoundaries, labelSP, numSP):
    print("Generating Superpixel adjacency matrix...")
    
    #Initialize adjacency matrix
    spAdjMatrix = np.zeros((numSP, numSP))
    
    #Extract image size attributes
    imgHeight, imgWidth = spBoundaries.shape[:2]
    
    #Iterate through image, altering spAdjMatrix for every boundary pixel
    for y in range(1, imgHeight - 1):
        for x in range(1, imgWidth - 1):
            #Skip iteration if not border pixel
            if (spBoundaries[y][x][0] == 0):
                continue
            
            #Add all adjacent SP to spAdjMatrix
            currLabel = labelSP[y][x]
            spAdjMatrix[currLabel][labelSP[y + 1][x]] = 1
            spAdjMatrix[currLabel][labelSP[y - 1][x]] = 1
            spAdjMatrix[currLabel][labelSP[y][x + 1]] = 1
            spAdjMatrix[currLabel][labelSP[y][x - 1]] = 1
            
    #Return completed adjacency matrix
    return spAdjMatrix
    
#Group Superpixels by the book they belong to
def groupSuperpixels(inputImg, ocrResults, xDiff, yDiff, spBoundaries, numSP, labelSP, labelColors, colorDistThresh=10, posThresh=3.5, maxSP=1500):
    #Initialize output array that holds four coordinates for each book
    bookCoords = []
    
    #Get Superpixel adjacency matrix
    spAdjMatrix = generateSuperpixelAdjacencyMatrix(spBoundaries, labelSP, numSP)
    
    #Determine Superpixels belonging to each book text detection
    for i in range(15, len(ocrResults)):
        #Assign initial Superpixels within detection box
        bookSP = assignInitialSuperpixels(np.array(ocrResults[i]), xDiff, yDiff, labelSP)
        print(len(bookSP))
        
        #Get book max and min pos
        bookPos = getBookSPPos(bookSP, labelColors)
        
        #Check whether to add adjacent Superpixels until none are added
        prevBookSPLen = 0
            
        #Check whether to add adjacent Superpixels
        for j in bookSP:
            #Skip execution if above max number of SPs
            if (len(bookSP) > maxSP):
                break
            
            for spID in range(numSP):
                #Skip iteration if not adjacent or already in bookSP
                if (spAdjMatrix[j][spID] == 0 or spID in bookSP):
                    continue
                    
                #Check if under color threshold
                colorDist = math.sqrt(pow((labelColors[j][0] - labelColors[spID][0]),2) + pow((labelColors[j][1] - labelColors[spID][1]),2) + pow((labelColors[j][2] - labelColors[spID][2]),2))
                if (colorDist > colorDistThresh):
                    continue
                
                #Add to bookSP if under position threshold for x or y
                xDist = bookPos[1] - bookPos[0]
                yDist = bookPos[3] - bookPos[2]
                if (np.abs(labelColors[spID][4] - bookPos[0]) < (posThresh * xDist)):
                    if (np.abs(labelColors[spID][5] - bookPos[1]) < (posThresh * xDist)):
                        bookSP.append(spID)
                        continue
                            
                if (np.abs(labelColors[spID][6] - bookPos[2]) < (posThresh * yDist)):
                    if (np.abs(labelColors[spID][7] - bookPos[3]) < (posThresh * yDist)):
                        bookSP.append(spID)
                    
        print(len(bookSP))
            
        #Get mask of book
        bookMask = getBookMask(inputImg, bookSP, labelSP)
        
        #Save mask
        plt.imsave(outPath + str(i) + ".jpg", bookMask)
    
    
    #return bookCoords
    
#Extract book spines
def extractBooks(inputImg, outPath, sigma=2, regionSize=20):
    '''#Make directory to save output images
    if (not os.path.exists(outPath)):
        os.mkdir(outPath)
        
    #Detect text in input image
    print("Detecting text regions...")
    ocrResults = findTextRegions(inputImg)
    
    #Cull bad detections
    print("Culling bad/repeat detections...")
    ocrResults = cullDetections(ocrResults)
    plt.imsave(outPath + 'Culled.jpg', drawDetections(inputImg, ocrResults))'''
        
    with open(outPath + "ocrCulled.pickle", "rb") as f:
        ocrResults = pkl.load(f)
    
    #Extract shelf
    print("Isolating bookshelf...")
    imgShelf, xDiff, yDiff = isolateShelf(inputImg, ocrResults)
    plt.imsave(outPath + 'Shelf.jpg', imgShelf)
    
    #Convert image to LAB for SLIC effectiveness
    imgLab = cv.cvtColor(imgShelf, cv.COLOR_BGR2Lab)
    
    #Apply Gaussian blur to image
    imgLab = cv.GaussianBlur(imgLab, (3,3), sigma)
    
    #Generate Superpixel object from image
    print("Generating Superpixel image...")
    objSP = cv.ximgproc.createSuperpixelSLIC(imgLab, cv.ximgproc.MSLIC, regionSize)
    
    objSP.iterate(10)
    spBoundaries = objSP.getLabelContourMask()
    print(objSP.getNumberOfSuperpixels())
    plt.imsave(outPath + 'Lab.jpg', imgLab)
    cv.imwrite(outPath + 'SPBound.jpg', spBoundaries)
    labelSP = objSP.getLabels()
    numSP = objSP.getNumberOfSuperpixels()
    
    with open(outPath + "superpixels.pickle", "wb") as f:
        pkl.dump(labelSP, f)
    
    #TO-DO: Replace 3217 with actual number variable
    print("Generating Superpixel visual...")
    imgSP, labelColors = visualizeSuperpixels(imgLab, labelSP, numSP)
    plt.imsave(outPath + 'SP.jpg', imgSP)
    plt.imsave(outPath + 'SPBox.jpg', drawDetectionsCropped(imgSP, ocrResults, xDiff, yDiff))
    
    with open(outPath + "labelColors.pickle", "wb") as f:
        pkl.dump(labelColors, f)
        
    #Get Superpixel boundary mask
    spBoundaries = cv.imread(outPath + 'SPBound.jpg')
    
    #Group Superpixels according to the book they belong to
    bookCoords = groupSuperpixels(imgLab, ocrResults, xDiff, yDiff, spBoundaries, numSP, labelSP, labelColors)
    
    '''#Get threshold for image
    threshold = otsuWrapper(imgLab[:,:,0])
    
    #Threshold image
    ret, imgThreshold = cv.threshold(imgLab[:,:,0], threshold, 255, cv.THRESH_BINARY)
    
    plt.imsave(outPath + 'thresh.jpg', imgThreshold)'''

#Get edges
def getBookEdges(inputImg, outPath):
    for i in range(0, 180, 10):
        edges = cv.Canny(inputImg, i, 180, L2gradient=True)
    
        plt.imsave(outPath + "edge" + str(i) + ".png", edges)
    
    
#Get paths for input and output files
srcDir = os.path.dirname(os.path.abspath(__file__))
inPath = str(srcDir + '\\' + sys.argv[1])
outPath = str(srcDir + '\\' + sys.argv[2])

#Import input image
inputImg = plt.imread(inPath)

#Apply pre-processing to input image, then save to outPath
extractBooks(inputImg, outPath)
#getBookEdges(inputImg, outPath)

#plt.imsave(outPath +'easyOCR.jpg', findTextRegions(inputImg, windowSize=50))