from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import math
import sys


# Turns quota enofrcement on and off
QuotaEnforcement = 0
# Number of clusters
seats = 3
#These settings should eventually be moved to command line arguments
data = pd.read_csv('./Ballots/' + sys.argv[1])
data.head()

def mult_ballots(l, weights):
    return np.concatenate([np.repeat(i[0], i[1]) for i in zip(l,weights)])

wb1 = mult_ballots(data.iloc[:, 2].values, data["Weight"].values)
wb2 = mult_ballots(data.iloc[:, 3].values, data["Weight"].values)
fs = list(zip(wb1, wb2))
for index in range(4, len(data.columns)):
    wb = mult_ballots(data.iloc[:, index].values, data["Weight"].values)
    fs = [x + (y,) for x, y in zip(fs, wb)]
ballots = np.array(fs, dtype=np.int8)

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, ord=2, axis=ax)


shape = ballots.shape

C = np.random.random_integers(np.min(ballots), np.max(ballots), (seats, shape[1]))

# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(ballots))

error = dist(C, C_old, None)

def quotas(votes, seats, type):
    qtypes = { 'hare': (len(votes)/seats), 
               'hvar': (len(votes)/seats)+1, #Hare variant that requires an extra vote; pretty sure someones come up with it already but I could not find documents on it. Seems to produce deterministic results in combination with "curr" quota enforcement.
              'droop': ((len(votes)/(seats+1))+1),  #Not really sure how to implement droop and hb quotas? I think that mathematically in this system we might want to multiply them by two. Otherwise they produece infinite loops.
                 'hb': (len(votes)/(seats+1)*2),
                'mod': (len(votes)/seats)+((len(votes)/(seats+1))/(seats+1))} #Random quota I came up with while playing around, in between hare and droop * 2
    return qtypes[type]

quota = quotas(ballots, seats, 'hare')

#Chooses the quota Enforcement rule. 
#Curr flips the value at the current indice, Rand just flips a random one.
#Not generalized to n seats, disabled currently
def quotaEnforcement(clusters, cluster, index, ruletype):
    rule = {"curr": index, "rand": np.random.randint(0, len(clusters))}
    while(clusters[rule["rand"]] != cluster): 
        rule["rand"] = np.random.randint(0, len(clusters))
    return rule[ruletype]

# Loop will run till the error becomes zero
while error != 0:
    print("Error amounts: ", error)
    # Assigning each value to its closest cluster
    for i in range(len(ballots)):
        distances = dist(ballots[i], C)
        cluster = np.argmin(distances)
        n = []
        if(len([n for n in range(len(clusters)) if clusters[n] == cluster]) > quota and QuotaEnforcement == 1):   #Quota enforcement. Not generalized to n-seats yet.
            cluster = 1-cluster
            #expellee = quotaEnforcement(clusters, cluster, i, "curr")   
            #clusters[expellee] = 1 - clusters[expellee]            
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(seats):
        points = [ballots[j] for j in range(len(ballots)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)

#calculate winners
for i in range(seats):
  points = [ballots[j] for j in range(len(ballots)) if clusters[j] == i]
  ranges = np.mean(points, axis=0)
  for index in range(2, len(data.columns)):
    print(data.iloc[:, index].name, ": ", ranges[index-2])
  print("Group members:", len(points))