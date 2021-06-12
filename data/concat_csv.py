import pandas as pd

df1 = pd.read_csv('data/new_dist_kpts1.csv', header=None)
df2 = pd.read_csv('data/new_dist_kpts2.csv', header=None)

df = pd.concat([df1, df2], ignore_index=True)
df.to_csv('data/point_dist_kpts.csv')
