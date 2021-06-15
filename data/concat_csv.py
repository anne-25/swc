import pandas as pd
import numpy as np
import sys

def concatnate(filename1, filename2):
    df1 = pd.read_csv(filename1, header=None)
    df2 = pd.read_csv(filename2, header=None)

    df = pd.concat([df1, df2], ignore_index=True)
    df = np.array(df)
    np.savetxt('data/new_dist_kpts.csv', df, delimiter=',')

if __name__ == '__main__':
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]

    concatnate(filename1, filename2)
