# python -m pip install opencv-python
import cv2 as cv
# python -m pip install numpy==1.19.3
import numpy as np
import glob
# python -m pip install tqdm
from tqdm import tqdm

# Calibration
chessboard_size = (4,7)

# Array definitions
obj_pointsL = []        # Real world space 3D points
img_pointsL = []        # Image plane 3D points
objpL = np.zeros((np.prod(chessboard_size),3),dtype=np.float32)
objpL[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

obj_pointsR = []        # Real world space 3D points
img_pointsR = []        # Image plane 3D points
objpR = np.zeros((np.prod(chessboard_size),3),dtype=np.float32)
objpR[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

# Read calibration image set
calibration_paths = glob.glob('../IMAGES/calib2/*.jpg')
FoundCornerCountL = 0
FoundCornerCountR = 0

# Find the chessboard corners
for image_path in tqdm(calibration_paths):
    # Load the image
    print(image_path)
    image = cv.imread(image_path)
    Y, X, Ch = image.shape
    # Split the image
    imageL = image[0:int(Y), 0:int(X/2)]
    imageR = image[0:int(Y), int(X/2):int(X)]
    # Convert to grayscale
    imageLg = cv.cvtColor(imageL, cv.COLOR_BGR2GRAY)
    imageRg = cv.cvtColor(imageR, cv.COLOR_BGR2GRAY)
    #debug
    # imageLg = cv.resize(imageLg, (1000,1000))
    # imageRg = cv.resize(imageRg, (1000,1000))
    # cv.imshow("Image L", imageLg)
    # cv.waitKey(1000)
    # cv.imshow("Image R", imageRg)
    # cv.waitKey(1000)
    # Process the images
    imageLg = cv.GaussianBlur(imageLg, (11,11), -2)
    imageLg = cv.addWeighted(imageLg, 1.5, imageLg, -0.5, 0, None)
    imageRg = cv.GaussianBlur(imageRg, (5,5), -2)
    imageRg = cv.addWeighted(imageRg, 1.5, imageRg, -0.5, 0, None)
    #debug
    # imageLg = cv.resize(imageLg, (1000,1000))
    # imageRg = cv.resize(imageRg, (1000,1000))
    # cv.imshow("Image L", imageLg)
    # cv.waitKey(1000)
    # cv.imshow("Image R", imageRg)
    # cv.waitKey(1000)
    retL, cornersL = cv.findChessboardCornersSB(imageLg, chessboard_size, None, None)
    cv.drawChessboardCorners(imageLg, chessboard_size, cornersL, retL)
    retR, cornersR = cv.findChessboardCornersSB(imageRg, chessboard_size, None, None)
    cv.drawChessboardCorners(imageRg, chessboard_size, cornersR, retR)
    #debug
    # imageLg = cv.resize(imageLg, (1000,1000))
    # imageRg = cv.resize(imageRg, (1000,1000))
    # cv.imshow("Image L", imageLg)
    # cv.waitKey(500)
    # cv.imshow("Image R", imageRg)
    # cv.waitKey(500)

    if retL == True:
        FoundCornerCountL += 1
    
    if retR == True:
        FoundCornerCountR += 1

    # If a chessboard is found, refine the corner area for subpixel accuracy
    if retL == True:
        print("Chessboard found in image left half.")
        # Define criteria
        criteriaL = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        cornersL2 = cv.cornerSubPix(imageLg, cornersL, (3,3), (-1,-1), criteriaL)
        obj_pointsL.append(objpL)
        img_pointsL.append(cornersL)
        # Update with new corners
        imageLg = cv.cvtColor(imageLg, cv.COLOR_GRAY2BGR)
        cv.drawChessboardCorners(imageLg, chessboard_size, cornersL2, retL)
        #debug
        imageLgdisp = cv.resize(imageLg, (1000,1000))
        cv.imshow('Image Left Half', imageLgdisp)
        cv.waitKey(250)
        imageLg = cv.cvtColor(imageLg, cv.COLOR_BGR2GRAY)

    if retR == True:
        print("Chessboard found in image right half.")
        # Define criteria
        criteriaR = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        cornersR2 = cv.cornerSubPix(imageRg, cornersR, (3,3), (-1,-1), criteriaR)
        obj_pointsR.append(objpR)
        img_pointsR.append(cornersR)
        # Update with new corners
        imageRg = cv.cvtColor(imageRg, cv.COLOR_GRAY2BGR)
        cv.drawChessboardCorners(imageRg, chessboard_size, cornersR2, retR)
        #debug
        imageRgdisp = cv.resize(imageRg, (1000,1000))
        cv.imshow('Image Right Half', imageRgdisp)
        cv.waitKey(250)
        imageRg = cv.cvtColor(imageRg, cv.COLOR_BGR2GRAY)

cv.destroyAllWindows()
# How many images yielded calibration inputs?
print(FoundCornerCountL)
print(FoundCornerCountR)

# Calibrate camera for each lens
retL, CmatrixL, distCoeffsL, rvecsL, tvecsL = cv.calibrateCamera(obj_pointsL, img_pointsL, imageLg.shape[::-1], None, None)
retR, CmatrixR, distCoeffsR, rvecsR, tvecsR = cv.calibrateCamera(obj_pointsR, img_pointsR, imageRg.shape[::-1], None, None)

# Export the calibration data (TODO)

# Output
print("Return L:" + str(retL))
print("Return R:" + str(retR))

print("Camera matrix L:")
print(CmatrixL)

print("Camera matrix R:")
print(CmatrixR)

FxL = CmatrixL[1,1]*23.5/(X)
print(FxL)
FxR = CmatrixR[1,1]*23.5/(X)
print(FxR)