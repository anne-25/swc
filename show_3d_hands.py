import numpy as np
import matplotlib.pyplot as plt
from calibrate import get_projection_matrix
from utils import DLT

def read_keypoints(filename):
    fin = open(filename, 'r')

    kpts = []
    while(True):
        line = fin.readline()
        if line == '': break

        line = line.split()
        line = [int(s) for s in line]

        line = np.reshape(line, (21, 2))
        kpts.append(line)

    kpts = np.array(kpts)
    return kpts

def visualize_3d(kpts_cam0, P0, kpts_cam1, P1):

    p3ds = []
    for frame0_kpts, frame1_kpts in zip(kpts_cam0, kpts_cam1):
        for uv1, uv2 in zip(frame0_kpts, frame1_kpts):
            if uv1[0] == -1 or uv2[0] == -1:
                _p3d = [-1, -1, -1]
            else:
                _p3d = DLT(P0, P1, uv1, uv2) #calculate 3d position of keypoint
            p3ds.append(_p3d)
    """This contains the 3d coordinates"""
    p3ds = np.reshape(p3ds, (-1, 21, 3))

    """Apply coordinate rotations to point z axis as up"""
    Rz = np.array(([[0., -1., 0.],
                    [1.,  0., 0.],
                    [0.,  0., 1.]]))
    Rx = np.array(([[1.,  0.,  0.],
                    [0., -1.,  0.],
                    [0.,  0., -1.]]))
    p3ds_rotated = []
    for frame in p3ds:
        frame_kpts_rotated = []
        for kpt in frame:
            if kpt[0] != -1 and kpt[1] != -1:
                kpt_rotated = Rx @ Rz @ kpt
            else:
                kpt_rotated = kpt
            frame_kpts_rotated.append(kpt_rotated)
        p3ds_rotated.append(frame_kpts_rotated)

    """this contains 3d points of each frame"""
    p3ds_rotated = np.array(p3ds_rotated)

    #save to file
    outf = open('data/test/test_3d.dat', 'w')
    for frame_kpts in p3ds_rotated:
        for kpt in frame_kpts:
            kpt = [str(k) for k in kpt]
            outf.write(kpt[0] + ',' + kpt[1] + ',' + kpt[2] + '\n')

    outf.close()

    """Now visualize in 3D"""

    thumb_f = [[0,1],[1,2],[2,3],[3,4]]
    index_f = [[0,5],[5,6],[6,7],[7,8]]
    middle_f = [[0,9],[9,10],[10,11],[11, 12]]
    ring_f = [[0,13],[13,14],[14,15],[15,16]]
    pinkie_f = [[0,17],[17,18],[18,19],[19,20]]
    fingers = [thumb_f, index_f, middle_f, ring_f, pinkie_f]
    fingers_colors = ['red', 'blue', 'green', 'black', 'orange']

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, kpts3d in enumerate(p3ds_rotated):
        if i%2 == 0: continue
        for finger, finger_color in zip(fingers, fingers_colors):
            for _c in finger:
                ax.plot(xs = [kpts3d[_c[0],0], kpts3d[_c[1],0]], ys = [kpts3d[_c[0],1], kpts3d[_c[1],1]], zs = [kpts3d[_c[0],2], kpts3d[_c[1],2]], linewidth = 4, c = finger_color)

        ax.set_xlim3d(-7, 8)
        ax.set_xlabel('x')
        ax.set_ylim3d(-7, 8)
        ax.set_ylabel('y')
        ax.set_zlim3d(0, 15)
        ax.set_zlabel('z')
        #plt.savefig('figs/fig_' + str(i) + '.png')
        plt.pause(0.01)
        ax.cla()


if __name__ == '__main__':

    kpts_cam0 = read_keypoints('data/test/kpts_cam0.dat')
    kpts_cam1 = read_keypoints('data/test/kpts_cam1.dat')

    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    visualize_3d(kpts_cam0, P0, kpts_cam1, P1)
