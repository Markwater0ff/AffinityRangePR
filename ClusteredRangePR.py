from copy import deepcopy
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
from scipy.spatial import distance
import math
import sys



fig = plt.figure()
ax = Axes3D(fig)

data = pd.read_csv('./Ballots/' + sys.argv[1])
#print(data.shape)
data.head()

#print(X)
def mult_ballots(l, weights):
  return np.concatenate([np.repeat(i[0], i[1]) for i in zip(l,weights)])

f1 = mult_ballots(data['Z'].values,data['Weight'].values)
f2 = mult_ballots(data['X'].values,data['Weight'].values)
f3 = mult_ballots(data['Q'].values,data['Weight'].values)

X = np.array(list(zip(f1, f2, f3)), dtype=np.int8)
ax.scatter(f1, f2, f3)
#plt.show()

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, ord=2, axis=ax)

# Number of clusters
k = 2

# X coordinates of random centroids
C_z = np.random.randint(0, np.max(X), size=k)
# Y coordinates of random centroids
C_x = np.random.randint(0, np.max(X), size=k)

C_q = np.random.randint(0, np.max(X), size=k)
C = np.array(list(zip(C_z, C_x, C_q)), dtype=np.int8)
#print(C)

# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))

error = dist(C, C_old, None)

def quotas(votes, seats, type):
    qtypes = { 'hare': (len(votes)/seats), 
               'hvar': (len(votes)/seats)+1, 
              'droop': ((len(votes)/(seats+1))+1), 
                 'hb': (len(votes)/(seats+1)*2),
                'mod': (len(votes)/seats)+((len(votes)/(seats+1))/(seats+1))}
    return qtypes[type]

quota = quotas(X, k, 'hvar')
print(quota)
print(len(X))

#Chooses the quota Enforcement rule. 
#Curr flips the value at the current indice, Rand just flips a random one.
def quotaEnforcement(clusters, cluster, index, ruletype):
    rule = {"curr": index, "rand": np.random.randint(0, len(clusters))}
    while(clusters[rule["rand"]] != cluster): 
        rule["rand"] = np.random.randint(0, len(clusters))
    return rule[ruletype]

# Loop will run till the error becomes zero
while error != 0:
    print(error)
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
        n = []
        if(len([n for n in range(len(clusters)) if clusters[n] == cluster]) > quota):
            expellee = quotaEnforcement(clusters, cluster, i, "curr")
            clusters[expellee] = 1 - clusters[expellee]
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)

#colors = ['r', 'g', 'b', 'y', 'c', 'm']
#fig = plt.figure()
#ax = Axes3D(fig)
#for i in range(k):
#        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
#        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=7, c=colors[i])
#ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', s=200, c='#050505')
#plt.show()

#ballot_clusters = list(zip(X,clusters))



#calculate winners
for i in range(k):
  points = [X[j] for j in range(len(X)) if clusters[j] == i]
  ranges = np.mean(points, axis=0)
  print("Z: ", ranges[0], " X: ", ranges[1], " Q:", ranges[2])
  print(len(points))

#print(points)
#print(X)

