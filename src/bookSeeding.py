import cv2 as cv
import easyocr
import numpy as np
import matplotlib.pyplot as plt

#Draw OCR detections on image
def drawDetections(img, ocrResults):
    #Copy image to annotate
    annotatedImg = img.copy()
    
    #Iterate through results and draw box for each
    for (bbox, _) in ocrResults:
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
def findTextRegions(inputImg, outPath):
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