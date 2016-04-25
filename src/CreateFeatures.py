# -*- coding: utf-8 -*-
"""
Created on Fri Apr 03 19:28:12 2015
Non Intrusive Load Monitoring for Energy Disaggregation for the REDD data
Class project for CS446: Machine Learning @ University of Illinois at Urbana-Champaign
REDD Reference: "J. Zico Kolter and Matthew J. Johnson.  REDD: A public data set for
energy disaggregation research.  In proceedings of the SustKDD
workshop on Data Mining Applications in Sustainability, 2011."

@authors: Anand Deshmukh, Danny Lohan
University of Illinois at Urbana-Champaign
"""

import numpy as np
import matplotlib.pyplot as plt 
import csv
import time
from scipy import interpolate
from MLData import createInstances, deviceErrors
from sklearn.naive_bayes import MultinomialNB 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.lda import LDA
from sklearn.ensemble import RandomForestClassifier
from energyCalcs import actDevEnergy,appDevEnergy,energyComp
from sklearn.cluster import KMeans

t0 = time.time()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
datapath = 'Data/house_3/'
weatherfile = 'Data/weather/20110405hourly_Boston.txt'
weatherfile_test = 'Data/weather/20110405hourly_Boston.txt'
fileprefix = 'channel_'

train_vals_start = 120001
train_vals_end = 240002

test_vals_start = 240002
test_vals_end = 240002 + 120001

# specify the timewindow for instances
timewindow = 90

# training data arrays
device_timer = np.zeros(shape=(train_vals_end - train_vals_start,9))
device_power = np.zeros(shape=(train_vals_end - train_vals_start,9))
total_device_power = np.zeros(shape=(train_vals_end - train_vals_start))
weather_timer = np.zeros(shape=(1980))
weather_temp = np.zeros(shape=(1980))
weather_data = np.zeros(shape=(train_vals_end - train_vals_start))

# test data arrays
device_timer_test = np.zeros(shape=(test_vals_end - test_vals_start,9))
device_power_test = np.zeros(shape=(test_vals_end - test_vals_start,9))
total_device_power_test = np.zeros(shape=(test_vals_end - test_vals_start))
weather_timer_test = np.zeros(shape=(1980))
weather_temp_test = np.zeros(shape=(1980))
weather_data_test = np.zeros(shape=(test_vals_end - test_vals_start))


# devices to be used in training and testing
use_idx = np.array([3,4,6,7,10,11,13,17,19])
uidx = 0

################################################################
## READ TRANING DATA ##
# read the weather data
wfile = open(weatherfile,'rt') 
rownum = 0
try:
    wreader = csv.reader(wfile, delimiter=',')
    for row in wreader:
        #print row[1]+','+row[2]+','+row[10]
        wdate = row[1]
        wtime = row[2]
        wdatelist = list(wdate)
        wtimelist = list(wtime)
        timedatestr = ''.join(wdatelist[0:4])+'-'+ ''.join(wdatelist[4:6])+'-'+''.join(wdatelist[6:8]) +'-'+ ''.join(wtimelist[0:2])+'-'+''.join(wtimelist[2:4])+'-'+'00'
        weather_timer[rownum] = int(time.mktime(time.strptime(timedatestr,"%Y-%m-%d-%H-%M-%S")))
        weather_temp[rownum] = int(row[10])
        #print str(weather_timer[rownum]) + ','+ str(weather_temp[rownum])
        rownum = rownum + 1
finally:
    wfile.close
    
################################################################
# read the device data
for device in range(0,20):
    channel = device + 3
    filename = 'channel_'+ str(channel) +'.dat'
    filepath = datapath + filename
    xtemp,ytemp = np.loadtxt(filepath,unpack=True)
    #plt.plot(device_timer[:,device],device_power[:,device])
    if (device in use_idx):
        device_timer[:,uidx] = xtemp[train_vals_start:train_vals_end]# - 1302930690
        device_power[:,uidx] = ytemp[train_vals_start:train_vals_end]
        total_device_power += ytemp[train_vals_start:train_vals_end]
        uidx = uidx + 1        
    
interp_func = interpolate.interp1d(weather_timer, weather_temp)    
weather_data = interp_func(device_timer[:,0])


################################################################
## READ TEST DATA ##
# read the weather data
uidx = 0
wfile = open(weatherfile_test,'rt') 
rownum = 0
try:
    wreader = csv.reader(wfile, delimiter=',')
    for row in wreader:
        #print row[1]+','+row[2]+','+row[10]
        wdate = row[1]
        wtime = row[2]
        wdatelist = list(wdate)
        wtimelist = list(wtime)
        timedatestr = ''.join(wdatelist[0:4])+'-'+ ''.join(wdatelist[4:6])+'-'+''.join(wdatelist[6:8]) +'-'+ ''.join(wtimelist[0:2])+'-'+''.join(wtimelist[2:4])+'-'+'00'
        weather_timer_test[rownum] = int(time.mktime(time.strptime(timedatestr,"%Y-%m-%d-%H-%M-%S")))
        weather_temp_test[rownum] = int(row[10])
        #print str(weather_timer[rownum]) + ','+ str(weather_temp[rownum])
        rownum = rownum + 1
finally:
    wfile.close
    
#################################################################
## read the device data
for device in range(0,20):
    channel = device + 3
    filename = 'channel_'+ str(channel) +'.dat'
    filepath = datapath + filename
    xtemp,ytemp = np.loadtxt(filepath,unpack=True)
    #plt.plot(device_timer[:,device],device_power[:,device])
    if (device in use_idx):
        device_timer_test[:,uidx] = xtemp[test_vals_start:test_vals_end]# - 1302930690
        device_power_test[:,uidx] = ytemp[test_vals_start:test_vals_end]
        total_device_power_test += ytemp[test_vals_start:test_vals_end]
        uidx = uidx + 1        
    
interp_func = interpolate.interp1d(weather_timer, weather_temp)    
weather_data = interp_func(device_timer[:,0])
#temp_array = range(train_vals_start,train_vals_end)

#plt.plot(temp_array,total_device_power)
################################################################
# create the instances and labels from the training data
classify = 1 # 1 - Naive Bayes, 2 - Regression, 3 - SVM, 4 - Linear Discriminant Analysis, 5 - Random Forest Classifier
train_instances,train_labels,train_labels_binary = createInstances(total_device_power, device_timer, device_power, weather_data,classify,timewindow)
test_instances,test_labels,test_labels_binary = createInstances(total_device_power_test, device_timer_test, device_power_test, weather_data_test,classify,timewindow)

t1 = time.time()

Ttime = t1-t0;
print 'Computational Time = ' + str(Ttime)  
np.save('data11', train_instances)
np.save('data12', train_labels)
np.save('data13', train_labels_binary)
np.save('data14', test_instances)
np.save('data15', test_labels)
np.save('data16', test_labels_binary)
np.save('data17', use_idx)
np.save('data18', device_power)
np.save('data19', device_timer)
np.save('data110', device_power_test)
np.save('data111', device_timer_test)

classify = 2 # 1 - Naive Bayes, 2 - Regression, 3 - SVM, 4 - Linear Discriminant Analysis, 5 - Random Forest Classifier
train_instances,train_labels,train_labels_binary = createInstances(total_device_power, device_timer, device_power, weather_data,classify,timewindow)
test_instances,test_labels,test_labels_binary = createInstances(total_device_power_test, device_timer_test, device_power_test, weather_data_test,classify,timewindow)

t2 = time.time()

Ttime = t2-t1;

print 'Computational Time = ' + str(Ttime)  
np.save('data21', train_instances)
np.save('data22', train_labels)
np.save('data23', train_labels_binary)
np.save('data24', test_instances)
np.save('data25', test_labels)
np.save('data26', test_labels_binary)
np.save('data27', use_idx)
np.save('data28', device_power)
np.save('data29', device_timer)
np.save('data210', device_power_test)
np.save('data211', device_timer_test)


##################
