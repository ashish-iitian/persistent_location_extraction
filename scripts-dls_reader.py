from operator import add, sub
from math import radians, cos, sin, asin, sqrt
from signal import signal, SIGPIPE, SIG_DFL 
import matplotlib
#set backend to PNG as headless server cannot use default GTK backend
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import numpy as np
from pyspark import SparkContext, SparkConf
from itertools import chain
import cPickle as pickle

def fn(lon1,lat1,lon2,lat2):
	"""
    	Calculate the great circle distance between two points 
    	on the earth (specified in decimal degrees)
    	"""
    	# convert decimal degrees to radians 
	lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
	dlon = lon2 - lon1 
	dlat = lat2 - lat1 
	a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
	c = 2 * asin(sqrt(a)) 
	r = 6371 # Radius of earth in kilometers. Use 3956 for miles
	return c * r * 1000 #in meters

def compute(distance, time):
	"""
	compute velocity given the distance and time values 
	and prevent zero division errors
	"""

	dt = sum(distance)
	t = sum(time)
	if(t == 0):
		return "user mobility info cannot be obtained"
	else:
		return float(dt/t)

def calc_min(x,y):
	"""
	compute min_dst by subtracting error radii
	"""
	sm = np.array(x)
	er = np.array(y)
	return list(sm-er)

def calc_max(x,y):
	"""
	compute max_dst by adding error radii
	"""
	sm = np.array(x)
	er = np.array(y)
	return list(sm+er)

signal(SIGPIPE, SIG_DFL)
conf = (SparkConf().set('spark.driver.maxResultSize','16G')).setAppName("dls_reader")
sc = SparkContext(conf = conf)
data = sc.textFile("/srv/cluster-vol0/michael/DLS_data_1_28/")
dt = data.map(lambda s: s.split(","))\
	.filter(lambda line: line[10] == 'CST')\
	.map(lambda token: (int(token[0]),token[3:8]))\
    .groupByKey()\
	.map(lambda x: (x[0], [(int(i[0]),int(i[1]),float(i[2]),float(i[3]),int(i[4])) for i in x[1]]))\
	.map(lambda x : (x[0], sorted(x[1])))\
	.collect()
ls = dt.map(lambda x: len(x[1]))\
	.collect()
mini = min(ls)
maxi = max(ls)
medi = np.median(ls)
meani = np.mean(ls)

#obtain the total time a user is active (range not helpful coz there are IDs with juz one entry and others with too large a number of instances and hence, duration)
sum_t = dt.map(lambda x: sum[(y[1][1]-y[1][0]) for y in x[1]])\
	.collect()
sum_time = list(chain.from_iterable(sum_t))

# obtain the longest duration for which a user used his cellular service, consider users with >= 2 new locations over 2 days
d = dt.filter(lambda x: len(x[1])>=2)
result_tm = d.map(lambda x: (x[0],max([x[1][i][1]-x[1][i][0] for i in range(len(x[1]))])))\
	.collect()
tm = d.map(lambda x: (x[0],[x[1][i+1][0]-x[1][i][0] for i in range(len(x[1])-1)]))\
	.collect()
mean_tm = [np.mean(tm[i][1]) for i in range(len(tm))]	
dst = d.map(lambda x : (x[0],[fn(x[1][i][3],x[1][i][2],x[1][i+1][3],x[1][i+1][2]) for i in range(len(x[1])-1)]))\
	.collect()
mean_dst = [np.mean(dst[i][1]) for i in range(len(dst))]	
error = d.map(lambda x: (x[0],[x[1][i][4]+x[1][i+1][4] for i in range(len(x[1])-1)]))\
	.collect()
min_dst = [(dst[i][0],calc_min(dst[i][1],error[i][1])) for i in range(len(dst))]
max_dst = [(dst[i][0],calc_max(dst[i][1],error[i][1])) for i in range(len(dst))]
vel = [compute(dst[i][1],tm[i][1]) for i in range(len(dst))]
min_vel = [compute(min_dst[i][1],tm[i][1]) for i in range(len(min_dst))]
max_vel = [compute(max_dst[i][1],tm[i][1]) for i in range(len(max_dst))]

#to study variation in max duration of activity
mind = min(result_tm)
maxd = max(result_tm)
med = np.median(result_tm)

print "no. of locations/user: mini= %d max= %d median= %f mean= %f\n" %(mini,maxi,medi,meani)
print "min duration: %d\n" %(mind)
print "max duration: %d\n" %(maxd)
print "median duration: %f" %(med)


plt.figure()
plt.hist(ls,bins=2500, range=(0,2500))
plt.title('Locations reported for a user in DLS_data_1_28')
plt.xlabel('No. of locations reported')
plt.ylabel('number of users')
plt.savefig("/home/akumar13/dls_1_28/location_plot.png")

plt.figure()
plt.hist(vel,bins=100, range=(0,100))
plt.title('avg vel. vs users for DLS_data_1_28')
plt.xlabel('avg. velocity of user in m/s')
plt.ylabel('number of users')
plt.savefig("/home/akumar13/dls_1_28/vel_plot.png")

plt.figure()
plt.hist(min_vel,bins=100, range=(0,100))
plt.title('min vel. vs users for DLS_data_1_28')
plt.xlabel('avg. min velocity of user in m/s')
plt.ylabel('number of users')
plt.savefig("/home/akumar13/dls_1_28/min_vel_plot.png")

plt.figure()
plt.hist(max_vel,bins=150, range=(0,150))
plt.title('max vel. vs users for DLS_data_1_28')
plt.xlabel('avg. max velocity of user in m/s')
plt.ylabel('number of users')
plt.savefig("/home/akumar13/dls_1_28/max_vel_plot.png")

plt.figure()
plt.hist(result_tm,bins=3000, range=(0,60000))
plt.title('max. Duration vs users for DLS_data_1_28')
plt.xlabel('max duration in seconds')
plt.ylabel('number of users')
plt.savefig("/home/akumar13/dls_1_28/duration_plot.png")

plt.figure()
plt.hist(sum_time,bins=100000, range=(0,100000))
plt.title('sum of Durations vs users for DLS_data_1_28')
plt.xlabel('sum of durations in seconds')
plt.ylabel('number of users')
plt.savefig("/home/akumar13/dls_1_28/sum_of_durations.png")

plt.figure()
plt.hist(mean_dst,bins=50000, range=(0,50000))
plt.title('Mean dst. travelled by users for DLS_data_1_28')
plt.xlabel('mean distance travelled in meters')
plt.ylabel('number of users')
plt.savefig("/home/akumar13/dls_1_28/mean_dst_travelled.png")

plt.figure()
plt.hist(mean_tm,bins=1000, range=(0,1000))
plt.title('Mean transit Duration vs users for DLS_data_1_28')
plt.xlabel('Mean transit durations in seconds')
plt.ylabel('number of users')
plt.savefig("/home/akumar13/dls_1_28/mean_transit_duration.png")

tm = [tm[i][1] for i in xrange(len(tm))]
tm_list = list(chain.from_iterable(tm))
plt.figure()
plt.hist(tm_list,bins=500, range=(0,1000))
plt.title('Duration vs No. of transitions for DLS_data_1_28')
plt.xlabel('Duration between transitions in seconds')
plt.ylabel('number of transitions')
plt.savefig("/home/akumar13/dls_1_28/transit_duration_plot.png")

plt.figure()
plt.hist(tm_list,bins=1000, range=(0,1000), normed=True, cumulative=True, fill=False)
plt.title('CDF for Duration between transition for DLS_data_1_28')
plt.xlabel('Duration between transitions in seconds')
plt.ylabel('fraction of transitions')
plt.grid(b=True, which='major', color='b', linestyle='-')
plt.grid(b=True, which='minor', color='b', linestyle='-')
plt.savefig("/home/akumar13/dls_1_28/transit_duration_cdf.png")

dst = [dst[i][1] for i in xrange(len(dst))]
dst_list = list(chain.from_iterable(dst))
#range_dst = np.max(dst_list) - np.min(dst_list)
plt.figure()
plt.hist(dst_list, bins= 50000, range=(0,50000))
plt.title('Dist. during transition vs No. of transitions for DLS_data_1_28')
plt.xlabel('Distance travelled during transitions in meters')
plt.ylabel('number of transitions')
plt.savefig("/home/akumar13/dls_1_28/transit_dist_plot.png")

plt.figure()
plt.hist(dst_list,bins=50000, range=(0,50000), fill=False, normed=True, cumulative=True)
plt.title('CDF for Dist. during transition for DLS_data_1_28')
plt.xlabel('Distance travelled during transitions in meters')
plt.ylabel('fraction of transitions')
plt.grid(b=True, which='major', color='b', linestyle='-')
plt.grid(b=True, which='minor', color='b', linestyle='-')
plt.savefig("/home/akumar13/dls_1_28/transit_dist_cdf.png")

min_dst = [min_dst[i][1] for i in xrange(len(min_dst))]
mind_list = list(chain.from_iterable(min_dst))
#num = np.max(mind_list) - np.min(mind_list)
plt.figure()
plt.hist(mind_list,bins=50000, range=(-25000,25000))
plt.title('Min Dist. during transition vs No. of transitions for DLS_data_1_28')
plt.xlabel('Min. Distance travelled during transitions in meters')
plt.ylabel('number of transitions')
plt.savefig("/home/akumar13/dls_1_28/transit_min_dist_plot.png")

plt.figure()
plt.hist(mind_list,bins=50000, range=(-25000,25000), fill=False, normed=True, cumulative=True)
plt.title('CDF for Min. Dist. during transitions for DLS_data_1_28')
plt.xlabel('Min. Dist. travelled during transitions in meters')
plt.ylabel('fraction of transitions')
plt.grid(b=True, which='major', color='b', linestyle='-')
plt.grid(b=True, which='minor', color='b', linestyle='-')
plt.savefig("/home/akumar13/dls_1_28/transit_min_dist_cdf.png")

max_dst = [max_dst[i][1] for i in xrange(len(max_dst))]
maxd_list = list(chain.from_iterable(max_dst))
#range_max = np.max(maxd_list) - np.min(maxd_list)
plt.figure()
plt.hist(maxd_list,bins=50000, range=(0,50000))
plt.title('Max Dist. during transition vs No. of transitions for DLS_data_1_28')
plt.xlabel('Max Distance travelled during transitions in meters')
plt.ylabel('number of transitions')
plt.savefig("/home/akumar13/dls_1_28/transit_max_dist_plot.png")

plt.figure()
plt.hist(maxd_list,bins=50000, range=(0,50000), fill=False, normed=True, cumulative=True)
plt.title('CDF for Max. Dist. during transitions for DLS_data_1_28')
plt.xlabel('Max. Dist travelled during transitions in meters')
plt.ylabel('fraction of transitions')
plt.grid(b=True, which='major', color='b', linestyle='-')
plt.grid(b=True, which='minor', color='b', linestyle='-')
plt.savefig("/home/akumar13/dls_1_28/transit_max_dist_cdf.png")