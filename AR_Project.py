import cv2
import numpy as np
import winsound

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

imgTarget = cv2.imread('dram_pic.jpeg')
# myVid = cv2.VideoCapture('sample_vid.webm')
myVid = cv2.VideoCapture('dram.mp4')

imgTarget = cv2.resize(imgTarget, (520, 500))

detection = False
frameCounter = 0

success, imgVideo = myVid.read()

# hT, wT, cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (520, 500))

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgTarget, None)
# imgTarget = cv2.drawKeypoints(imgTarget,kp1, None)

while True:
    success, imgWebCam = cap.read()
    imgAug = imgWebCam.copy()
    kp2, des2 = orb.detectAndCompute(imgWebCam, None)
    # imgWebCam = cv2.drawKeypoints(imgWebCam, kp2, None)

    if not detection:
        myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0
        winsound.PlaySound(None, winsound.SND_PURGE)
    else:
        if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):
            myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
            winsound.PlaySound(None, winsound.SND_PURGE)
        elif frameCounter == 1:
            winsound.PlaySound("nsksi.wav", winsound.SND_ASYNC | winsound.SND_ALIAS)
        success, imgVideo = myVid.read()
        imgVideo = cv2.resize(imgVideo, (520, 500))

    bf = cv2.BFMatcher()
    if np.any(des1) or np.any(des2):
        matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))
    imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebCam, kp2, good, None, flags=2)

    if len(good) > 15:
        detection = True
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
        if np.any(matrix):
            pts = np.float32([[0, 0], [0, 500], [520, 500], [520, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            img2 = cv2.polylines(imgWebCam, [np.int32(dst)], True, (255, 0, 255), 3)

            imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebCam.shape[1], imgWebCam.shape[0]))

            maskNew = np.zeros((imgWebCam.shape[0], imgWebCam.shape[1]), np.uint8)
            cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
            maskInverse = cv2.bitwise_not(maskNew)

            imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInverse)
            imgAug = cv2.bitwise_or(imgWarp, imgAug)

            cv2.imshow('Webcam Target', imgAug)

        print(matrix)


    else:
        detection = False
        imgAug = 0
    #     cv2.imshow('Image Warp', imgWarp)
    #     cv2.imshow('Image 2', img2)
    # cv2.imshow('Image Features', imgFeatures)
    # cv2.imshow('Image Target', imgTarget)
    # cv2.imshow('Video Target', imgVideo)
    if np.any(imgAug):
        pass
    else:
        cv2.imshow('Webcam Target', imgWebCam)
    frameCounter += 1
    q = cv2.waitKey(20) & 0xff
    if q == 27:
        break

cap.release()
cv2.destroyAllWindows()
