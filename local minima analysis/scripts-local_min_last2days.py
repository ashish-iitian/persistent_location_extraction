import csv
import operator
import cPickle as pickle
from math import radians, cos, sin, asin, sqrt, floor, ceil
from signal import signal, SIGPIPE, SIG_DFL 
import matplotlib
#set backend to PNG as headless server cannot use default GTK backend
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import numpy as np
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

def bin_and_return_minima(user_dls,start):
    """bins data and returns estimated local minima
    """
    end = MAX_HOURS=48
    hrs = [0]*MAX_HOURS
    for x in user_dls:
        #create unique locations/hour
        for i in range((x[0]-start)//3600,((x[1]-start)//3600)+1):
            if i<MAX_HOURS:
                hrs[i] += 1
    #find hour of greatest mobility
    indx,val = max(enumerate(hrs[:24]), key = operator.itemgetter(1))
    if(indx <= 12):
		if(indx >= 4):
			ind = 0
		else:
			return [-1,hrs]
    else:
        ind = indx-12
	if(indx+12 < MAX_HOURS):
		end = indx+12
	else:
		if(MAX_HOURS-indx >= 4):
			end = MAX_HOURS
		else:
			return [-1,hrs]
    if(all(v== 0 for v in hrs[ind:indx]) or all(v== 0 for v in hrs[indx:end])):
        #if no data 12 hours before and/or after the max mobility, then discard
        return [-1,hrs]
    else:
        #find minimum 12 hours before max
        val_1 = min(hrs[i] for i in range(ind,indx) if hrs[i] > 0)
        ind_1 = hrs[ind:indx].index(val_1)
        #find minimum 12 hours after max
        val_2 = min(hrs[i] for i in range(indx,end) if hrs[i] > 0)
        ind_2 = hrs[indx:end].index(val_2)
        return [ind+ind_1,indx+ind_2]

def persistance_dist(user_data, st):
    """Returns the distance between two locations estimated to be persistent
    """
    start=st
    ret=bin_and_return_minima(user_data[1],start)
    if (-1 in ret):
        return None
    lat_long=[]
    for tm in ret:
        for i in user_data[1]:
            if((i[0]-start)//3600 <= tm <= (i[1]-start)//3600):
                lat_long.append([i[3],i[2],(i[0]+i[1])/2])
                break
    return (ll_dist(lat_long[0][0],lat_long[0][1],lat_long[1][0],lat_long[1][1]),abs(lat_long[0][2]-lat_long[1][2]))


def plot_time_dist_spread(time_diff,loc_diff,time_bin=3600,loc_bin=500,time_max=86400, loc_max=10000):
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
    plt.savefig("/home/akumar13/local_min_time_dist_last2days.png")


def main():
	signal(SIGPIPE, SIG_DFL)
	conf = (SparkConf().set('spark.driver.maxResultSize','30G')).setAppName("local_min")
	sc = SparkContext(conf = conf)
	end = 1419321599
	start = end - 86400*2
    # data is a list of tuples, each tuple having a user-id and list of tuples (each tuple being the 4 params)
	TEST=True
	if TEST:
		f = open('/home/akumar13/data_sampled_15000.pk','rb')
		data = pickle.load(f)
		f.close()
		f = open('/srv/cluster-vol0/michael/scratch/dls_users_10000.pk','rb')
		dat = pickle.load(f)
		f.close()
		for x in dat:
			data.append(x)
		data_rdd=sc.parallelize(data)
	else:
		data_rdd = sc.textFile("/srv/cluster-vol0/michael/DLS_data_1_28/")
		data_rdd = data_rdd.map(lambda line: line.split(','))\
			.filter(lambda tokens: tokens[3]>="start" and tokens[4]<= "end")\
			.map(lambda tokens: (tokens[0],(int(tokens[3]),int(tokens[4]),float(tokens[5]),float(tokens[6]))))\
			.groupByKey()

	persistant_locs=data_rdd.map(lambda kv: persistance_dist(kv,start)).filter(lambda x: x is not None).collect()
	loc_diff=[]
	time_diff=[]
	for i in persistant_locs:
		loc_diff.append(i[0])
		time_diff.append(i[1])
	print data_rdd.count()
	print len(loc_diff)
	plot_time_dist_spread(time_diff, loc_diff)

if __name__=="__main__":
	main()
