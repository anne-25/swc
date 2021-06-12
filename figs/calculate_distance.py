import numpy as np
import csv

def read_keypoints(filename):
    fin = open(filename, 'r')

    kpts = []
    while(True):
        line = fin.readline()
        if line == '': break

        line = line.split()
        line = [int(s) for s in line]
        if line[0] == -1: continue

        line = np.reshape(line, (21, 2))
        kpts.append(line)

    kpts = np.array(kpts)
    return kpts

def calculate_distance(kpts):
    dist = []
    d = []
    for points in kpts:
        for index, p1 in enumerate(points):
            for i in range(index + 1, len(points)):
                u = p1 - points[i]
                u = np.linalg.norm(u)
                d.append(u)
        dist.append(d)

    dist = np.array(dist)
    return dist

def write_csv(filename, num):
    with open(f'dist_{num}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(filename)

if __name__ == '__main__':

    kpts_cam0 = read_keypoints('kpts_cam0.dat')
    kpts_cam1 = read_keypoints('kpts_cam1.dat')

    kpts_cam0 = calculate_distance(kpts_cam0)
    kpts_cam1 = calculate_distance(kpts_cam1
    )

    write_csv(kpts_cam0, 0)
    write_csv(kpts_cam1, 1)
