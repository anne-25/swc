import numpy as np
import sys


def read_file(filename):
    with open(filename, 'r') as f:
        dat = []
        while True:
            line = f.readline()
            if line == '': break

            line = line.split()
            l = [float(s) for s in line]
            dat.append(l)

    dat = np.array(dat)

    return dat

def cut_preprocess(old_dat, length):
    new_dat = [old_dat[i] for i in range(len(old_dat) - length)]

    outf = open('test/new_dist_kpts.csv', 'w')
    for dist in new_dat:
        for i in range(len(dist) - 1):
            outf.write(str(dist[i]) + ',')
        outf.write(str(dist[-1]) + '\n')
    outf.close()




if __name__ == '__main__':
    filename = sys.argv[1]
    length = int(sys.argv[2])
    dat = read_file(filename)
    cut_preprocess(dat, length)
    pass
