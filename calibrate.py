import cv2 as cv
import glob
import numpy as np
import datetime
from utils import Annotation, _make_homogeneous_rep_matrix, DLT
#import matplotlib.pyplot as plt

camera_id_to_loc = {0:4, 1:10}

#text settings
font = cv.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (0, 0, 255)
thickness = 2
frame_shape = [720, 1280]

def save_frames_four_cams(savefolder, count = 36):

    cap0 = cv.VideoCapture(camera_id_to_loc[0])
    cap1 = cv.VideoCapture(camera_id_to_loc[1])

    caps = [cap0, cap1]

    for cap in caps:
        cap.set(3, 1280)
        cap.set(4, 720)

    frame_counter = 300
    frame_iter = 0
    while True:
        _, frame0 = cap0.read()
        _, frame1 = cap1.read()

        frame0_cop = frame0.copy()
        frame1_cop = frame1.copy()

        #crop to 720x720 from center of frame
        frame0_cop = frame0_cop[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
        frame1_cop = frame1_cop[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]

        frame0_cop = cv.resize(frame0_cop, (500, 500))
        frame1_cop = cv.resize(frame1_cop, (500, 500))

        frame0_cop = cv.putText(frame0_cop, str(frame_iter) + ' : ' + str(frame_counter), org, font, fontScale, color, thickness, cv.LINE_AA)
        frame_counter -= 1

        if frame_counter == 0:
            frame_counter = 150
            cv.imwrite(savefolder + 'frame_c0_' + str(frame_iter) + '.png', frame0)
            cv.imwrite(savefolder + 'frame_c1_' + str(frame_iter) + '.png', frame1)
            frame_iter += 1
            #break

        cv.imshow('frame0', frame0_cop)
        cv.imshow('frame1', frame1_cop)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_iter == count:
            break

    for cap in caps:
        cap.release()
        cv.destroyAllWindows()

def calibrate_camera(images_folder):
    images_names = glob.glob(images_folder)
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        im = im[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
        images.append(im)

    # for i, im in enumerate(images):
    #     cv.imshow('im' + str(i), im)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = 5 #number of checkerboard rows.
    columns = 8 #number of checkerboard columns.
    world_scaling = 1. #change this to the real world square size. Or not.

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space


    for i, frame in enumerate(images):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:

            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            cv.imshow('img', frame)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints.append(corners)


    cv.destroyAllWindows()
    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print(images_folder)
    print('rmse:', ret)
    print('camera matrix:\n', cmtx)
    print('distortion coeffs:', dist)
    #print('Rs:\n', rvecs)
    #print('Ts:\n', tvecs)

    return cmtx, dist


def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder_c1, frames_folder_c2):
    #read the synched frames
    c1_images_names = sorted(glob.glob(frames_folder_c1))
    c2_images_names = sorted(glob.glob(frames_folder_c2))

    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        _im = _im[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
        c1_images.append(_im)

        _im = cv.imread(im2, 1)
        _im = _im[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
        c2_images.append(_im)


    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    rows = 5 #number of checkerboard rows.
    columns = 8 #number of checkerboard columns.
    world_scaling = 1. #change this to the real world square size. Or not.

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space

    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (5, 8), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (5, 8), None)

        if c_ret1 == True and c_ret2 == True:

            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            p0_c1 = corners1[0,0]
            p0_c2 = corners2[0,0]

            #frame1 = cv.putText(frame1, 'O', (p0_c1[0], p0_c1[1]), font, fontScale, color, thickness, cv.LINE_AA)
            cv.drawChessboardCorners(frame1, (rows,columns), corners1, c_ret1)
            cv.imshow('img', frame1)

            #frame2 = cv.putText(frame2, 'O', (p0_c2[0], p0_c2[1]), font, fontScale, color, thickness, cv.LINE_AA)
            cv.drawChessboardCorners(frame2, (rows,columns), corners2, c_ret2)
            cv.imshow('img2', frame2)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)

    print(ret)
    return R, T

def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_id):

    outf = open('camera_parameters/c' + str(camera_id) + '.dat', 'w')

    outf.write('intrinsic:\n')
    for l in camera_matrix:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('distortion:\n')
    for en in distortion_coefs[0]:
        outf.write(str(en) + ' ')
    outf.write('\n')

def save_camera_to_camera_coord_transforms(rot, trans, camera_id):

    outf = open('camera_parameters/relative_transform_c' + str(camera_id) + '.dat', 'w')

    outf.write('R:\n')
    for l in rot:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in trans:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.close()

def read_camera_to_camera_coord_transforms(camera_id):

    inf = open('camera_parameters/relative_transform_c' + str(camera_id) + '.dat', 'r')

    inf.readline()
    rot = []
    trans = []
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        rot.append(line)

    inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        trans.append(line)

    inf.close()
    return np.array(rot), np.array(trans)

def read_camera_parameters(camera_id):

    inf = open('camera_parameters/c' + str(camera_id) + '.dat', 'r')

    cmtx = []
    dist = []

    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        cmtx.append(line)

    line = inf.readline()
    line = inf.readline().split()
    line = [float(en) for en in line]
    dist.append(line)

    return np.array(cmtx), np.array(dist)

def get_world_space_origin(cmtx, dist, img_path):

    frame = cv.imread(img_path, 1)
    frame = frame[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = 5 #number of checkerboard rows.
    columns = 8 #number of checkerboard columns.
    world_scaling = 1. #change this to the real world square size. Or not.

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = frame.shape[1]
    height = frame.shape[0]

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

    #frame1 = cv.putText(frame, 'O', (corners[0,0,0], corners[0,0,1]), font, fontScale, color, thickness, cv.LINE_AA)
    cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
    cv.imshow('img', frame)
    cv.waitKey(0)

    ret, rvec, tvec = cv.solvePnP(objp, corners, cmtx, dist)
    rvec, _  = cv.Rodrigues(rvec)

    return rvec, tvec

def save_world_space_origin(rot, trans, savefolder):

    outf = open(savefolder + 'c0_world_space_transform.dat', 'w')

    outf.write('R:\n')
    for l in rot:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in trans:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.close()

def read_world_space_origin(savefolder = 'camera_parameters/'):

    inf = open(savefolder + 'c0_world_space_transform.dat', 'r')

    inf.readline()
    rot = []
    trans = []
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        rot.append(line)

    inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        trans.append(line)

    inf.close()
    return np.array(rot), np.array(trans)

def _convert_to_homogeneous(pts):
    pts = np.array(pts)
    if len(pts.shape) > 1:
        w = np.ones((pts.shape[0], 1))
        return np.concatenate([pts, w], axis = 1)
    else:
        return np.concatenate([pts, [1]], axis = 0)

def get_projection_matrix(camera_id):

    #read camera parameters
    cmtx, dist = read_camera_parameters(camera_id)
    rvec0, tvec0 = read_world_space_origin()
    if camera_id != 0:
        rvec, tvec = read_camera_to_camera_coord_transforms(camera_id)
    else:
        rvec, tvec = None, None

    #calculate projection matrix
    if camera_id == 0:
        P = cmtx @ _make_homogeneous_rep_matrix(rvec0, tvec0)[:3,:]
    else:
        P = cmtx @ (_make_homogeneous_rep_matrix(rvec, tvec) @ _make_homogeneous_rep_matrix(rvec0, tvec0))[:3,:]

    #print(P)
    return P

if __name__ == '__main__':

    """save stereo calibration frames"""
    # save_frames_four_cams('calibration_frames/', 31)
    # quit()

    """calibrate to get intrinsic and distortion parameters of single cameras and save to file"""
    # for i in range(2):
    #     cmtx, dist = calibrate_camera('calibration_frames/frame_c' + str(i) + '*')
    #     #save to file
    #     save_camera_intrinsics(cmtx, dist, i)
    # quit()

    """stereo_calibrate."""
    # camera_matrices = []
    # distortion_coefs = []
    # for i in range(2):
    #     cmtx, dist = read_camera_parameters(i)
    #     camera_matrices.append(cmtx)
    #     distortion_coefs.append(dist)
    #
    # rot, trans = stereo_calibrate(camera_matrices[0], distortion_coefs[0], camera_matrices[1], distortion_coefs[1], 'calibration_frames/frame_c0*', 'calibration_frames/frame_c1*')
    # save_camera_to_camera_coord_transforms(rot, trans, i)
    # quit()

    """choose world space origin"""
    cmtx, dist = read_camera_parameters(0)
    rvec, tvec = get_world_space_origin(cmtx, dist, 'calibration_frames/frame_c0_30.png')
    save_world_space_origin(rvec, tvec, 'camera_parameters/')
    quit()

    #get_projection_matrix(1)

    pass
