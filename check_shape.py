import numpy as np
import sys
import os

def open_file(filename):

    inf = open(filename, 'r')

    dat = []
    base, ext = os.path.splitext(filename)
    if ext == '.csv':
        while True:
            line = inf.readline()
            if line == '': break

            line = line.split(',')
            line = [float(number) for number in line]
            if line[0] == -1: continue
            dat.append(line)
    else:
        while True:
            line = inf.readline()
            if line == '': break

            line = line.split()
            line = [float(number) for number in line]
            if line[0] == -1: continue
            dat.append(line)

    dat = np.array(dat)

    return dat

if __name__ == '__main__':
    filename = sys.argv[1]
    dat = open_file(filename)
    print(dat.shape)
