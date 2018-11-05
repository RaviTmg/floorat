import numpy as np
import cv2
import math
from PIL import Image
import os 

counter = 0
for root, directory, files in os.walk('test/input/'):
    for img in files:
        im = Image.open(os.path.join(root,img))
        image = np.array(im)
        print('Original Dimensions : ',image.shape)
        height = 640
        (h,w) = image.shape[:2]
        r = height / float(h)
        width = int(w*r)
        dim = (width, height)
        # resize image
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        print('Resized Dimensions : ',resized.shape)
        grayscale = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        sigma = 0.33
        v = np.median(resized)
        # apply automatic Canny edge detection using the computed median
        # lower = int(max(0, (1.0 - sigma) * v))
        # upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(grayscale, 100, 150, apertureSize=3, L2gradient=True)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 150) 
        lines1 = []
        for i in range(len(lines)):
            for j in range(len(lines)):
                if (i == j):continue
                if (abs(lines[i][0][1] - lines[j][0][1]) == 0):          
                    #You've found a parallel line!
                    lines1.append(lines[i])
                    lines1.append(lines[j])

        for line in lines1:
            rho, theta= line[0]
            # Stores the value of cos(theta) in a 
            a = np.cos(theta) 
            # Stores the value of sin(theta) in b 
            b = np.sin(theta) 
            # x0 stores the value rcos(theta) 
            x0 = a*rho
            # y0 stores the value rsin(theta) 
            y0 = b*rho
            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta)) 
            x1 = int(x0 + width*(-b)) 
            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta)) 
            y1 = (y0 + width*(a)) 
            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta)) 
            x2 = int(x0 - width*(-b)) 
            # y2 stores the rounded off value of (rsin(theta)-1000cos(theta)) 
            y2 = (y0 - width*(a)) 
            # find the slop of each line
            slope = (y2-y1)/(x2-x1)
            #dictionary of lines
            # cv2.line draws a line in img from the point(x1,y1) to (x2,y2). 
            # (0,255,0) denotes the colour of the line to be  
            #drawn. In this case, it is green.  
            if slope != float('-Inf'):
                cv2.line(resized,(x1,int(y1)), (x2,int(y2)), (0,255,0),1)  

        filename = "test/output/"+ os.path.splitext(img)[0]+".jpg"
        cv2.imwrite(filename, resized)