from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import sys
import pandas as pd
import math
from sklearn import preprocessing 
import argparse

parser = argparse.ArgumentParser(description='Run proportional election using range ballots and K-means grouping.')
parser.add_argument('Ballots', help='Name of ballots file which contains election data, place in the ballots folder. Must be csv format.')
parser.add_argument('Seats', help='Number of seats to run the election for.', type=int)
parser.add_argument('-a', action='store_true', help='Print affinities between each voter (optional)')

args = vars(parser.parse_args())
ballots_file = args['Ballots']
seats = args['Seats']
print_affinities = args['a']


pdata = pd.read_csv('./Ballots/' + ballots_file)
pdata.head()

def mult_ballots(l, weights):
    return np.concatenate([np.repeat(i[0], i[1]) for i in zip(l,weights)])

wb1 = mult_ballots(pdata.iloc[:, 2].values, pdata["Weight"].values)
wb2 = mult_ballots(pdata.iloc[:, 3].values, pdata["Weight"].values)
fs = list(zip(wb1, wb2))
for index in range(4, len(pdata.columns)):
    wb = mult_ballots(pdata.iloc[:, index].values, pdata["Weight"].values)
    fs = [x + (y,) for x, y in zip(fs, wb)]
ballots = np.array(fs, dtype=np.int8)

def cluster_sizes(seats, labels):
  sizes = []      
  for k in range(seats):
    sizes.append(cluster_size(seats, labels, k))
  return sizes

def cluster_size(seats, labels, k):
  return(len([i for i in labels if i == k]))

def largest_cluster(seats, labels):
  return max(cluster_sizes(seats, labels))

def quotas(votes, seats, type):
    qtypes = { 'hare': math.ceil(len(votes)/seats), 'hvar': (len(votes)/seats), 'droop': (((len(votes)/(seats+1))+1)*2), 'hb': (len(votes)/(seats+1)*2), 'mod': (len(votes)/seats)+((len(votes)/(seats+1))/(seats+1))} 
    return qtypes[type]

def above_quota_clusters(seats, labels,quota):
  sizes = cluster_sizes(seats, labels)
  return [ind for ind, i in enumerate(sizes) if i > quota]

def points_label_indxs(distances, cent_indx):
  dist_indxs, dist_points = [], []
  for ind, point in enumerate(distances):
    if labels[ind] == cent_indx:
      dist_indxs.append(ind)
      pointm = np.ma.array(point, mask=False)
      for i in above_quota_clusters(seats, labels, quota-1):
        pointm.mask[i] = True
      dist_points.append(pointm[np.argmin(pointm)]-point[cent_indx])
  return dist_indxs,dist_points

quota = quotas(ballots, seats, 'hare')

kmeans = KMeans(n_clusters=seats, algorithm="full", n_init=100, tol=0)  
kmeans.fit(ballots)

centroids = kmeans.cluster_centers_    
labels = kmeans.labels_
distances = kmeans.transform(ballots)

for cent_indx in above_quota_clusters(seats, labels,quota):
  while cluster_size(seats, labels, cent_indx) > quota:
    winner = np.argmax(centroids[cent_indx])
    dist_indxs, dist_points = points_label_indxs(distances, cent_indx)
    leastd_ncluster = np.argmin(dist_points)
    curr_distances = np.ma.array(list(distances[dist_indxs[leastd_ncluster]]), mask=False)
    for i in above_quota_clusters(seats, labels,quota-1):
      curr_distances.mask[i] = True
    next_closest_cluster_label = np.argmin(curr_distances)
    labels[[dist_indxs[leastd_ncluster]]] = next_closest_cluster_label
    

for i in range(seats):
  points = [ballots[j] for j in range(len(ballots)) if labels[j] == i]
  centroids[i] = np.mean(points, axis=0)

def ensure_most_preferred(group, winner, centroids):
  for comp_group in centroids:
    if comp_group[winner] > group[winner] and np.argmax(comp_group) == winner:
      winner = np.argsort(group, axis=0)[-2]
  return winner
    
  
for ind, group in enumerate(centroids):
  winner = ensure_most_preferred(group, np.argmax(group), centroids)
  print("Winner ", ind + 1, " is ", pdata.iloc[:, winner+2].name)
  print("\t", "Score:", group[winner])
  print("\t", "Group members: ", end="")
  for pind, name in enumerate(pdata['Ballot ID'].values):
    if labels[pind] == ind:
      print(name, " | ", end="")
  print()

def dist(a, b):
    return np.linalg.norm(a - b, ord=2, axis=None)

if(print_affinities == 1):
  print("Affinities: ")
  user_diffs = []
  users = pdata['Ballot ID'].values
  for i, main_user in enumerate(ballots):
    print(users[i], " - ", end="\t")
    for n, comp_user in enumerate(ballots):
      if n != i:
        print(users[n], ": ", "%.2f" % dist(main_user, comp_user), end="\t")
    print()