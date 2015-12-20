import csv
import operator
import cPickle as pickle
from scipy import stats
from math import radians, cos, sin, asin, sqrt, floor, ceil
from signal import signal, SIGPIPE, SIG_DFL 
import matplotlib
#set backend to PNG as headless server cannot use default GTK backend
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from pyspark.mllib.clustering import KMeans, KMeansModel
import numpy as np
from pyspark import SparkContext, SparkConf
import time


def fn(lon1,lat1,lon2,lat2):
	'''
    	Calculate the great circle distance between two points 
    	on the earth (specified in decimal degrees)
    '''
	lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
	dlon = lon2 - lon1 
	dlat = lat2 - lat1 
	a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
	c = 2 * asin(sqrt(a)) 
	r = 6371 # Radius of earth in kilometers. Use 3956 for miles
	return c * r * 1000 #in meters

def clusterer(input):

	clusters = KMeans.train(input,2,maxIterations=50,runs=50,initializationMode="random")	
	ls = clusters.clusterCenters
	return ls

def converter(number, start):
	''' 
	function that converts epoch time within a pd of two days into hour values in range(0,48)
	'''
	return (number-start)/3600.0

signal(SIGPIPE, SIG_DFL)
conf = (SparkConf().set('spark.driver.maxResultSize','30G')).setAppName("kmeans")
sc = SparkContext(conf = conf)
data = sorted(data,reverse = True, key = lambda data: len(data[1]))
d = data[0:500]
start = 1418889600
	
f = open('/home/akumar13/data_sampled_15000.pk','rb')
data = pickle.load(f)
# data is a list of tuples, each tuple having a user-id and list of tuples (each tuple being the 4 params)
f.close()

fh = open('/home/akumar13/dls_users_10000.pk','rb')
dat = pickle.load(fh)
fh.close()
for x in dat:
	data.append(x)

dt = sc.parallelize(d)\
	.map(lambda x: np.array([[converter(i[0],start),i[2],i[3]] for i in x[1]]))\
	.collect()
k = 0
cntr = [[] for x in xrange(0,500)]
spread = [0.0]*500
count = 0
for i in dt:
	cntr[k] = clusterer(sc.parallelize(i))
	spread[k] = fn(cntr[k][0][2],cntr[k][0][1],cntr[k][1][2],cntr[k][1][1])
	if(spread[k]<10000):
		count += 1
	k += 1
        print k
plt.figure()
plt.hist(spread, bins = range(0,10000,100))
plt.title('location spread obtained using k-means clustering')
plt.xlabel('separation between center of clusters in meters')
plt.ylabel('count')
plt.savefig("/home/akumar13/dls_1_28/kmeans_select.png")

plt.figure()
plt.hist(spread, bins = range(0,10000),normed = True, cumulative=True, fill = False)
plt.title('cdf of location spread obtained using k-means clustering')
plt.xlabel('separation between cluster centers in meters')
plt.ylabel('fraction of total count')
plt.grid(b=True, which='major', color='b', linestyle='-')
plt.grid(b=True, which='minor', color='b', linestyle='-')
plt.savefig("/home/akumar13/dls_1_28/kmeans_select_cdf.png")

with open('/home/akumar13/kmeans_select_spread.csv','w+') as f:
	writer = csv.writer(f)
	writer.writerow(spread)
f.close()

