import numpy as np
#import matplotlib.pyplot as plt
import cv2 as cv

def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P

def DLT(P1, P2, point1, point2):

    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    #print('A: ')
    #print(A)

    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)

    print('Triangulated point: ')
    print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]


class Annotation:

    #parameters used to draw the Rectangle
    mouse_down = False

    num_keypoints = 21

    edges = [[0,1],[1,2],[2,3],[3,4],
             [0,5],[5,6],[6,7],[7,8],
             [0,9],[9,10],[10,11],[11, 12],
             [0,13],[13,14],[14,15],[15,16],
             [0,17],[17,18],[18,19],[19,20]
            ]

    #indices to connect
    thumb_f = [[0,1],[1,2],[2,3],[3,4]]
    index_f = [[0,5],[5,6],[6,7],[7,8]]
    middle_f = [[0,9],[9,10],[10,11],[11, 12]]
    ring_f = [[0,13],[13,14],[14,15],[15,16]]
    pinkie_f = [[0,17],[17,18],[18,19],[19,20]]
    fingers = [thumb_f, index_f, middle_f, ring_f, pinkie_f]
    fingers_colors = [(0,0,0), (0,0,255), (255,0,0), (255,20,147), (0,255,0)]

    def __init__(self, img):
        self.img = img
        self.img_original = self.img.copy()
        self.saved_coords = []
        self.kpt_id = 0
        self._x = -1
        self._y = -1

    def Getkpts(self):
        while(len(self.saved_coords)) != self.num_keypoints:
            self.saved_coords = []
            self.img = self.img_original.copy()
            self._promptForCrop('Click body part')

        self._draw_stick_figure()
        return self.saved_coords


    def _promptForCrop(self, text = ''):

        cv.namedWindow('frame')
        cv.setMouseCallback('frame', self.onrelease)

        while(1):
            cv.imshow('frame', self.img)
            k = cv.waitKey(20) & 0xFF
            if k == 27:
                break

            if len(self.saved_coords) == self.num_keypoints:
                break

        cv.destroyAllWindows()


    def _draw_stick_figure(self):

        for finger, color in zip(self.fingers, self.fingers_colors):
            for _c in finger:
                cv.line(self.img, (self.saved_coords[_c[0]][0], self.saved_coords[_c[0]][1]),
                                  (self.saved_coords[_c[1]][0], self.saved_coords[_c[1]][1]), color, 1)

        cv.imshow('frame', self.img)
        while True:
            k = cv.waitKey(1000) # change the value from the original 0 (wait forever) to something appropriate
            if k == 27:
                break
            if cv.getWindowProperty('frame', cv.WND_PROP_VISIBLE) < 1:
                break

        cv.destroyAllWindows()


    def onrelease(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONUP:
            cv.circle(self.img,(x,y), 1,(255,0,0),-1)
            self.saved_coords.append([x,y])


if __name__ == '__main__':
    img = cv.imread('testing/frame_c0_0.png', 1)
    anno = Annotation(img)
    kpts = anno.Getkpts()
    print(kpts)
