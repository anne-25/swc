import numpy as np
import sys


def calculate_distance(dat):

    #calculate features
    dist = []
    for kpts in dat:
        d = []
        #distance between palms and fingertips, and between fingertips
        points = [kpts[0], kpts[4], kpts[8], kpts[12], kpts[16], kpts[20]]
        for idx, p1 in enumerate(points):
            if idx == 5: break
            for i in range(idx + 1, len(points)):
                u = p1 - points[i]
                u = np.linalg.norm(u)
                d.append(u)
        #distance between fingertip and third joint
        u = kpts[2] - kpts[4]
        u = np.linalg.norm(u)
        d.append(u)
        for i in range(4):
            u = kpts[5 + (i * 4)] - kpts[8 + (i * 4)]
            u = np.linalg.norm(u)
            d.append(u)
        dist.append(d)
        # print(dist)
    dist = np.array(dist)
    #save to file
    outf = open('test/dist_kpts.dat', 'w')
    for frame_dist in dist:
        for i in range(len(frame_dist)-1) :
            outf.write(str(frame_dist[i]) + ' ')
        outf.write(str(frame_dist[i + 1]) + '\n')
    outf.close()


def open_file(filename):

    inf = open(filename, 'r')

    dat = []
    while True:
        line = inf.readline()
        if line == '': break

        line = line.split(',')
        line = [float(number) for number in line]
        if line[0] == -1: continue
        dat.append(line)

    dat = np.array(dat)

    return dat

if __name__ == '__main__':

    filename = sys.argv[1]
    dat = open_file(filename)
    dat = dat.reshape((-1, 21, 3))
    calculate_distance(dat)
