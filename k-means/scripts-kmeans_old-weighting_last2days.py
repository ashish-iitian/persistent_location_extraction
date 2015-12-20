import csv
import operator
from scipy.cluster.vq import *
import cPickle as pickle
from math import radians, cos, sin, asin, sqrt, floor, ceil
from signal import signal, SIGPIPE, SIG_DFL 
import matplotlib
#set backend to PNG as headless server cannot use default GTK backend
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import numpy as np
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
	function that converts epoch time within a pd of two days into hour values in range(0,48)
	'''
	hour = (number-start)/3600.0
	return hour%24

def filterer(ls):
	"""filter data to only retain ones relevant to persistent location determination (atleast last longer than 5 minutes)"""
	lt = []
	for x in ls:
		if(x[1]-x[0] > 300):
			lt.append(x)
		else:
			pass
	return lt
	
def cluster_reduction(x):
#cluster data into 3 groups first, then extract the two tightest groups and combine them under the #assumption that these are likely the two persistent location groups. thereby reducing 3 cluster problem #into a 2 cluster one: persistent group and mobility group.

	cntr, label = kmeans2(x,3,iter= 100,minit='points')
	count = [0]*3
	score = [0.0]*3
	dist = [[] for i in range(3)]
# Try to come up with a score to represent tightness of each cluster (used Std Dev. to calculate scores)
	for i in xrange(0,len(x)):
		dist[label[i]].append(np.linalg.norm(cntr[label[i]]-x[i]))
		count[label[i]] += 1
	for i in range(0,3):
		score[i] = np.std(np.array(dist[i]))
	temp1 = min(enumerate(score), key = operator.itemgetter(1))[0]
	del score[temp1]
	temp2 = min(enumerate(score), key = operator.itemgetter(1))[0]
#obtained the two clusters we want to club into one by folding the time-axis about the mid for the clusters
	mid = (cntr[temp1][0]+cntr[temp2][0])/2.0
	for i in x:
		i[0] = i[0] % mid
	centroid = kmeans2(x,2,iter= 100,minit='points')[0]
	return centroid
		
def plot_time_dist_spread(time_diff,loc_diff,time_bin=3600,loc_bin=500,time_max=86400*2, loc_max=10000):
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
	plt.savefig("/home/akumar13/k-means_time_dist_last2days.png")
	
def main():
	signal(SIGPIPE, SIG_DFL)
	conf = (SparkConf().set('spark.driver.maxResultSize','30G')).setAppName("kmeans")
	sc = SparkContext(conf = conf)
	end = 1419321599
	start = end - 86400*2

# choose the active/mobile users through "filterer" fn() to apply k-means

	TEST=False
	if TEST:
		f = open('/home/akumar13/data_sampled_15000.pk','rb')
		data = pickle.load(f)
		f.close()
		f = open('/srv/cluster-vol0/michael/scratch/dls_users_10000.pk','rb')
		dat = pickle.load(f)
		f.close()
		for x in dat:
			data.append(x)
		data = sorted(data,reverse = True, key = lambda data: len(data))			
		data_rdd=sc.parallelize(data).map(lambda x: [x[0],filterer(x[1])])
	else:
		data_rdd = sc.textFile("/srv/cluster-vol0/michael/DLS_data_1_28/")
		data_rdd = data_rdd.map(lambda line: line.split(','))\
			.map(lambda tokens: (tokens[0],(int(tokens[3]),int(tokens[4]),float(tokens[5]),float(tokens[6]))))\
			.filter(lambda tokens: tokens[3]>="start" and tokens[4]<= "end")\
			.groupByKey().map(lambda x: (x[0],filterer(x[1])))

# the data is wrapped around in 24 hour length, meaning 25th hour collapses onto the 1st hour.
# weighing of data records is done on the basis of duration: times an entry duration exceeds 5 minutes #which becomes the times that entry gets replicated to apply logic of weights in k-means. K-means returns # 2 3-d values each being center of their clusters and we pass the lat-long parts of the 3 coordinates to # ll_dist() function.

	data = data_rdd.collect()
	for x in data:
		pilot = sc.parallelize(x[1]).map(lambda x: ((x[2],x[3]),list(x[0:2]))).countByKey()
		weight_dict = dict(pilot)
		ls_new = [i*(3*weight_dict[(i[2],i[3])]+ 2*(i[1]-i[0])/300) for i in x[1]]
		x = ls_new

	dt = sc.parallelize(data)\
		.map(lambda x: [[converter(i[0],start),i[2],i[3]] for i in x[1]])\
		.filter(lambda x: len(x)>2)\
		.map(lambda x: np.array(x))\
		.map(lambda x: cluster_reduction(x))

	spread = dt.map(lambda x: ll_dist(x[0][2],x[0][1],x[1][2],x[1][1])).collect()
	time_diff = dt.map(lambda x: int((abs(x[0][0]-x[1][0]))*3600)).collect()

	count = len([x for x in spread if x>10000])
	count2 = len([x for x in spread if x>2000])
	print "\n %d %d %d\n" %(count, count2, dt.count())
	with open('/home/akumar13/kmeans_weighted_scipy.csv','w+') as f:
		writer = csv.writer(f)
		writer.writerow(spread)
		writer.writerow(time_diff)
	f.close()
	plot_time_dist_spread(time_diff, spread)
	

if __name__ == "__main__":
	main()
