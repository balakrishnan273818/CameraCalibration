# -*- coding: utf-8 -*-
"""
Created on Fri Oct 05 17:03:22 2018

@author: balakrishnan
To run this script
camera calibration images ( checker board images) are required
which can be utilized from 
"C:\opencv\sources\samples\cpp\" folder should the occassion occur
"""

import cv2
print(cv2.__version__)
import numpy as np
import glob

#termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)

#to generate object points in a (x,y,z) fashion
objp = np.zeros((6*7,3), np.float32)# 6 rows, 7 columns and 3 dimensional data
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
#print(objp)

#arrays to store object points and image points from all images
objpoints = [] # 3d point in the real world space
imgpoints = [] # 2d point in the image plane

# to read anything and everything with extension '*jpg'
images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)#simply reading
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#converting to gray
    #finding and returning the corners of the chessboard
    ret,corners = cv2.findChessboardCorners(gray, (7,6), None)
    #if found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        #calculation sub pixal accuracy
        imgpoints.append(corners)
        
        #draw and display the corners
        cv2.drawChessboardCorners(img,(7,6),corners,ret)
       # cv2.imshow('img',img)
       # cv2.waitKey(500)
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                   gray.shape[::-1],None,None)
#ret = return value
#mtx = camera matrix
#dist = distortion coefficient
#rvecs = rotational vectors
#tvecs = translational vectors

print(ret)
print(mtx)
print(dist)
#print(rvecs)
#print(tvecs)

img = cv2.imread('left12.jpg')
h, w = img.shape[:2]
newcameramtx , roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
newcameramtx1 , roi1 = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

#Method - 01
#undistort
dst = cv2.undistort(img , mtx, dist, None,newcameramtx)

dst1 = cv2.undistort(img, mtx, dist, None, newcameramtx1)
#crop the image
x,y,w,h = roi1
dst1 = dst1[y:y+h,x:x+w]
cv2.imwrite('calibresult.png',dst)
cv2.imwrite('calibresult1.png',dst1)

#Method - 02
#undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx1,(w,h),5)
dst2 = cv2.remap(img,mapx,mapy, cv2.INTER_LINEAR)

#crop the image
x,y,w,h = roi1
dst2 = dst2[y:y+h, x: x+w]
cv2.imwrite('calibresult_anothermethod.png',dst2)

np.savez
np.savetxt

mean_error = 0
#tot_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    print error
    mean_error += error
    
print ("total error: ", float(mean_error)/len(objpoints))
