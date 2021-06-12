import cv2 as cv

for i in range(14):
    cap = cv.VideoCapture(i)

    while True:
        ret, image = cap.read()

        if not ret: break

        cv.imshow('frame', image)
        k = cv.waitKey(1)

        print(i)

        if k == 27: break
