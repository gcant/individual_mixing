import individual_mixing
import numpy as np

g = np.genfromtxt('example_data/groups.txt').astype(int)
k = np.genfromtxt('example_data/edges.txt')

DM = individual_mixing.DirichletModel(k,g)
DM.fit()

