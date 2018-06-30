

## DESCRIPTION

This is a Python implementation, and proof of concept, for my idea for using data clustering algorithms as a proportional electoral system, essentially grouping people by the simialarity of their ballots and running an election for each seat within those affinity groups. 

## HOW TO USE

Currently there are two required paramaters, and one optional.

Python AffinityRangePR.py [Ballots_file] [Seats] [-a]

* Ballots_file - Here, type the name of a CSV file in the ./ballots directory containing the election data. Several test election files are included.

* Seats - The number of candidates that will be ultimately be elected, and the number of k-means clusters.

* -a - Optional paramater, if flagged then a printout of each voters distance to all other voters will be run at the end of the process. The lower this number, the closer the two ballots are, and the more similiar the two voters were to each other. Literally, this is euclidean distance, with each candidate being treated as a dimension, and the scores for the candidates representing a point in n-dimensional space.

## ELECTION FILE FORMAT

1. All columns besides the first two should simply be named after a candidate/option, and contain that ballots score for said candidate.

2. The first column in this file should be named "Ballot ID", and contain some kind of identifying value, like a name or a code. Mainly used while printing out affinities, if you don't care about that it can be filled with gibberish or even left blank. The core election code processes are based on index, rather than this value.

3. The second column should be named "Weight". If this value is greater than 1, it duplicates that ballot N times. Mainly used to quickly produce theoretical test elections.

## MISC

This election system solves some problems with other proportional range voting methods, it does not punish minority factions that are similiar to a majority faction as harshly for instances (they will tend to be clustered into their own group if large enough). 

It probably introduces a whole host of other problems too though, it still needs examination. This is just a proof of concept.

This is a non-deterministic process, as k-means does rely on a random initial seed. But I've found the scikit implementation to be very stable in its results, especially when told to re-run the process many times. I have yet to have it produce different winners due to randomness, but my dataset so far has been fairly limited.

There are other grouping methods, such as Guassian Mixture, that sometimes produce more reliable results, and I initially intended to expand the project to them. However, after giving it some thought, I found that the behavior of K-means is actually ideal. Most fancier grouping algorithms are concerned with keeping together shapes in the data, but here all we really care about is minimizing each voters average distance from the mean ballot of their group. If the ballots position in the "space" of this election happened to, by pure chance, resemble a couple of (n-dimensional) pieces of fruit, we probably don't really care if the algorithm has to break these shapes up in order to minimize average distance.

I altered the data to reassign people (based on who is closest to another cluster center) after the scikit k-means has run if any clusters have more than a Hare quota worth of members (i.e., I am enforcing equal cluster size). My implementation is pretty primitive I think, it can probably be improved upon, but it works for the time being.
