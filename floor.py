import numpy as np
import cv2
import math

img = cv2.imread("download.png")
height, width = img.shape[:2]
grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(grayscale, 75, 150, apertureSize=3, L2gradient=True)
lines = cv2.HoughLines(edges, 1, np.pi/180, 150 , 1)

for line in lines:
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
    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2). 
    # (0,255,0) denotes the colour of the line to be  
    #drawn. In this case, it is green.  
    cv2.line(img,(x1,int(y1)), (x2,int(y2)), (0,255,0),2)  

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

