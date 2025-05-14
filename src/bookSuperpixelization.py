import cv2 as cv
import math
import numpy as np
import matplotlib.pyplot as plt

#Function to crop contents of bounding box
def cropBook(img, bookMask):
    #Extract image size attributes
    imgHeight, imgWidth = img.shape[:2]
    
    #Convert bookMask to binary threshold
    bookBGR = cv.cvtColor(bookMask, cv.COLOR_Lab2BGR)
    bookGray = cv.cvtColor(bookBGR, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(bookGray, 65, 255, cv.THRESH_BINARY)
    
    #Find single contour for book
    contour, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, 2)
    contour = np.vstack(contour).squeeze()
    
    #Find rectangle to contain contour
    rect = cv.minAreaRect(contour)
    
    #Get the rotation angle of the box
    angle = rect[2]
    
    #Calculate the rotation matrix
    rotMat = cv.getRotationMatrix2D((imgWidth / 2, imgHeight / 2), angle, 1)
    
    #Rotate the image
    imgRot = cv.warpAffine(img, rotMat, (imgWidth, imgHeight))
    
    #Get the box corners
    bbox = cv.boxPoints(rect)
    
    #Transform according to the rotation matrix
    bboxRot = np.int64(cv.transform(np.array([bbox]), rotMat))[0]
    
    #Crop the rotated image
    x1, y1 = np.min(bboxRot, axis=0)
    x2, y2 = np.max(bboxRot, axis=0)
    imgCrop = imgRot[y1:y2, x1:x2]
    
    #Return cropped image
    return imgCrop

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
            labelCurr = labelSP[int(yLow + y)][int(xLow + x)]
            
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
            if (spBoundaries[y][x] == 0):
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
def groupSuperpixels(inputImg, outPath, ocrResults, xDiff, yDiff, spBoundaries, numSP, labelSP, labelColors, colorDistThresh=10, posThresh=3.5, maxSP=1500): 
    #Declare array to hold output images
    bookImgs = []
    
    #Get Superpixel adjacency matrix
    spAdjMatrix = generateSuperpixelAdjacencyMatrix(spBoundaries, labelSP, numSP)
    
    #Determine Superpixels belonging to each book text detection
    for i in range(len(ocrResults)):
        #Assign initial Superpixels within detection box
        bookSP = assignInitialSuperpixels(np.array(ocrResults[i]), xDiff, yDiff, labelSP)
        print(len(bookSP))
        
        #Get book max and min pos
        bookPos = getBookSPPos(bookSP, labelColors)
        
        #Flag whether to save
        sFlag = True
            
        #Check whether to add adjacent Superpixels
        for j in bookSP:
            #Skip execution if above max number of SPs
            if (len(bookSP) > maxSP):
                sFlag = False
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
                        bookPos = getBookSPPos(bookSP, labelColors)
                        continue
                            
                if (np.abs(labelColors[spID][6] - bookPos[2]) < (posThresh * yDist)):
                    if (np.abs(labelColors[spID][7] - bookPos[3]) < (posThresh * yDist)):
                        bookSP.append(spID)
                        bookPos = getBookSPPos(bookSP, labelColors)
                    
        print(len(bookSP))
        
        if (not sFlag):
            continue
            
        #Get mask of book
        bookMask = getBookMask(inputImg, bookSP, labelSP)
        
        #Crop bookMask
        bookCrop = cropBook(inputImg, bookMask)
        if (bookCrop.size > 0):
            plt.imsave(outPath + str(i) + ".jpg", bookCrop)
        
        #Append to output array
        bookImgs.append(bookCrop)
    
    #Return segmented books
    return bookImgs