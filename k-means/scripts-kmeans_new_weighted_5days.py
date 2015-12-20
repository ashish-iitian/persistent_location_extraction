import csv
import operator
import itertools
from compiler.ast import flatten
from scipy.cluster.vq import *
import cPickle as pickle
from math import radians, cos, sin, asin, sqrt, floor, ceil, pow
from signal import signal, SIGPIPE, SIG_DFL 
import matplotlib
#set backend to PNG as headless server cannot use default GTK backend
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from pyspark.mllib.clustering import KMeans, KMeansModel
import numpy as np
from numpy.linalg import matrix_rank
import numpy.ma as ma
from pyspark import SparkContext, SparkConf
import time

def ll_dist(lon1,lat1,lon2,lat2):
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

def converter(number, start):
	''' 
	function that converts epoch time within a pd of 5 days into hour values in range(0,24)
	'''
	hour = (number-start)/3600.0
	return hour%24

def filterer(ls):
	"""filter data to only retain ones relevant to persistent location determination (atleast last
	longer than 5 minutes)"""
	lt = []
	for x in ls:
		if(x[1]-x[0] > 300):
			lt.append(x)
		else:
			pass	
	return lt
	
def clusterer(input,no_of_clusters):
#cluster data into 3 groups first, then extract the two tightest groups and combine them under the #assumption that these are likely the two persistent location groups. thereby reducing 3 cluster problem #into a 2 cluster one: persistent group and mobility group.

	try:
		cntr, label = kmeans2(np.array(input),no_of_clusters,iter= 100,minit='points')
		count = 0
		score = [0.0]*no_of_clusters
		dist = [[] for i in range(no_of_clusters)]
	# Try to come up with a score to represent tightness of each cluster (used Std Dev. to calculate scores)
		for i in xrange(0,len(input)):
			cluster_id = label[i]
			dist[cluster_id].append(np.linalg.norm(cntr[cluster_id]-input[i]))
		for i in range(0,no_of_clusters):
			if(len(dist[i]) == 0):
				return None
			else:
				score[i] = sqrt(sum([val*val for val in dist[i]])/len(dist[i]))	
		if(no_of_clusters == 3):
			temp1 = min(enumerate(score), key = operator.itemgetter(1))[0]	
			del score[temp1]
			temp2 = min(enumerate(score), key = operator.itemgetter(1))[0]
		#obtained the two clusters we are interested in
			mid = (cntr[temp1][0]+cntr[temp2][0])/2.0
			adjusted_input = [[(x[0]-mid)%24,x[1],x[2]] for x in input]
			return adjusted_input
		else:
			spread = 0.0
			time_diff = 0
			loc_cluster_id = min(enumerate(score), key = operator.itemgetter(1))[0]
			loc_coordinates = cntr[loc_cluster_id]
			for j in xrange(0,len(input)):
				if(label[j] == loc_cluster_id):
					spread += pow(ll_dist(loc_coordinates[2],loc_coordinates[1],input[j][2],input[j][1]), 2)
					time_diff += pow(abs(loc_coordinates[0]-input[j][0]), 2)
					count += 1
			spread = sqrt(spread/count)
			time_diff = sqrt(time_diff/count)
			return ((loc_coordinates[1],loc_coordinates[2]),spread,time_diff)
	except ValueError:
		pass
		
		
def plot_time_dist_spread(time_diff,loc_diff,time_bin=3600,loc_bin=500,time_max=43200, loc_max=10000):
	"""Creates plot with 2d hist of time vs distance distribution
    Includes histograms and CDF's of delta time and distance
	"""
	loc_diff=np.array(loc_diff)
	time_diff=np.array(time_diff)

	time_diff=time_diff/3600.
	time_bin=time_bin/3600.
	time_max=time_max/3600.
	H, x, y = np.histogram2d(time_diff, loc_diff, bins=[np.arange(0,time_max,time_bin),np.arange(0,loc_max,loc_bin)])
	H=np.ma.masked_where(H<=0,H)
	plt.figure(figsize=(10,14),dpi=600)
	#combined 2D histogram
	plt.subplot2grid((3,2),(0,0),colspan=2)
	plt.pcolor(y,x,H)
	plt.title('Differences Between Calculated Persistent Locations')
	plt.ylabel('Time Difference (hours)')
	plt.xlabel('Location Difference (m)')
	C=plt.colorbar()
	C.set_label('Users per {0} min. per {1}m'.format(time_bin*60,loc_bin))
	plt.grid()
	plt.axis('tight')
    #location distribution
	plt.subplot2grid((3,2),(1,0))
	plt.hist(loc_diff, bins = np.arange(0,loc_max,loc_bin))
	plt.title('delta(location) Distribution')
	plt.xlabel('Location Difference (m)')
	plt.ylabel('count/{0}m'.format(loc_bin))
	#location CDF
	plt.subplot2grid((3,2),(2,0))
	plt.hist(loc_diff, bins = np.arange(0,loc_max,loc_bin),normed = True, cumulative=True, fill = False)
	plt.title('delta(location) CDF')
	plt.xlabel('Location Difference (m)')
	plt.ylabel('Fraction')
	#time distribution
	plt.subplot2grid((3,2),(1,1))
	plt.hist(time_diff, bins = np.arange(0,time_max,time_bin))
	plt.title('delta(time) Distribution')
	plt.xlabel('Time Difference (hours)')
	plt.ylabel('count/{0} min.'.format(time_bin*60))
	#time CDF
	plt.subplot2grid((3,2),(2,1))
	plt.hist(time_diff, bins = np.arange(0,time_max,time_bin), normed = True, cumulative=True, fill = False)
	plt.title('delta(time) CDF')
	plt.xlabel('Time Difference (hours)')
	plt.ylabel('Fraction')
	plt.savefig("/home/akumar13/k-means_time_dist_for_all.png")

def processor(list_oflists_oflists):
	x = flatten(list_oflists_oflists)
#	print "\n incoming rank %d\n" %matrix_rank(x)
	list_oflists = [x[i:i+5] for i in xrange(0,len(x),5)]
#	print "\n changed rank %d \n" %matrix_rank(list_oflists)
	return list_oflists	
	
def main():
	signal(SIGPIPE, SIG_DFL)
	conf = (SparkConf().set('spark.driver.maxResultSize','50G')).setAppName("kmeans_cut")
	sc = SparkContext(conf = conf)
	start = 1418889600

# having obtained a large list of users to work on, choose the most active/mobile users to apply k-means
	data_rdd = sc.textFile("/srv/cluster-vol0/michael/DLS_data_1_28/")
	data = data_rdd.map(lambda line: line.split(','))\
		.map(lambda tokens: (tokens[0],(int(tokens[3]),int(tokens[4]),float(tokens[5]),float(tokens[6]),int(tokens[9]))))\
		.groupByKey()
	cntr_label = data.filter(lambda x: len(x[1])>40)\
		.map(lambda x: (x[0],filterer(x[1])))\
		.map(lambda x: [[i for num in range(0,(i[4]/2+(i[1]-i[0])/100),5)] for i in x[1]])\
		.map(lambda x: processor(x))\
		.map(lambda x: [[converter(i[0],start),i[2],i[3]] for i in x])\
		.map(lambda x: clusterer(x,3))\
		.map(lambda x: clusterer(x,2))\
		.filter(lambda x: x is not None)\
		.collect()
	persistent_loc = [x[0] for x in cntr_label]
	spread = [x[1] for x in cntr_label]
	time_diff = [x[2]*3600 for x in cntr_label]
	total = data.count()
	compatible_users = len(cntr_label)
	count1 = len([x for x in spread if x>10000])
	count2 = len([x for x in spread if x>2000])
	print "\n %d %d %d %d\n" %(count1, count2, total, compatible_users)
	plot_time_dist_spread(time_diff, spread)
	
	with open('/home/akumar13/kmeans_weighted_3-2_cluster.csv','w+') as f:
		writer = csv.writer(f)
		writer.writerow(spread)
		writer.writerow(time_diff)
	f.close()

if __name__ == "__main__":
	main()
