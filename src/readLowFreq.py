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


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
datapath = 'Data/house_3/'
weatherfile = 'Data/weather/20110405hourly_Boston.txt'
weatherfile_test = 'Data/weather/20110405hourly_Boston.txt'
fileprefix = 'channel_'

train_vals_start = 0
train_vals_end = 120001

test_vals_start = 120001
test_vals_end = 240002

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
classify = 3 # 1 - Naive Bayes, 2 - Regression, 3 - SVM, 4 - Linear Discriminant Analysis, 5 - Random Forest Classifier
train_instances,train_labels,train_labels_binary = createInstances(total_device_power, device_timer, device_power, weather_data,classify,timewindow)
test_instances,test_labels,test_labels_binary = createInstances(total_device_power_test, device_timer_test, device_power_test, weather_data_test,classify,timewindow)

np.save()

for i in range (1,7): 
    classify = i
    if classify == 1:
        cLabel = 'Naive Bayes'
        clf = MultinomialNB()
        MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    
    elif classify == 2:
        cLabel = 'Logistic Regression'
        clf = LogisticRegression()
        LogisticRegression(C = 1.0, penalty = 'l1', tol=1e-6)
    
    elif classify == 3:
        cLabel = 'SVM'
        clf = SVC()
    
    elif classify == 4:
        cLabel = 'Linear Discriminant Analysis'
        clf = LDA()
    
    elif classify == 5:
        cLabel = 'Random Forest Classifier'
        clf = RandomForestClassifier(n_estimators=10)
        SVR(C = 1.0, epsilon=0.2)
    
    elif classify ==6:
        cLabel = 'K-means clustering'
        clf = KMeans(n_clusters=512, init='random')
        

    t0 = time.time()
    clf.fit(train_instances, train_labels)   
    t1 = time.time()
    nd = len(use_idx)
    
    # prediction on training and test data
    accuracyTr, dev_acc_train, predicted_labels_binary_train = deviceErrors(clf,nd,train_instances,train_labels,train_labels_binary)
    accuracyTs, dev_acc_test, predicted_labels_binary_test = deviceErrors(clf,nd,test_instances,test_labels,test_labels_binary)
    
    # prediction of device energy consumption
    agg_energy_train = train_instances[:,5]
    actEnergy_train = actDevEnergy(device_power,device_timer,nd)
    appEnergy_train = appDevEnergy(train_labels_binary,agg_energy_train,nd)
    preEnergy_train = appDevEnergy(predicted_labels_binary_train,agg_energy_train,nd)

    acTap_train, acTpre_train, apTde_train = energyComp(actEnergy_train, appEnergy_train, preEnergy_train)


    agg_energy_test = test_instances[:,5]
    actEnergy_test = actDevEnergy(device_power_test,device_timer_test,nd)
    appEnergy_test = appDevEnergy(test_labels_binary,agg_energy_test,nd)
    preEnergy_test = appDevEnergy(predicted_labels_binary_test,agg_energy_test,nd)
  
    acTap_test, acTpre_test, apTde_test = energyComp(actEnergy_test, appEnergy_test, preEnergy_test)

  
#    t2 = time.time()
#    t3 = time.time()

    trainTime = t1-t0
    test1Time = t2-t1
    test2Time = t3-t2
    print '================================================================================'   
    print 'Classifier = ' + cLabel
    print 'Computational Expense for Training Classifier = ' + str(trainTime)  + 's'
    print '------------------------- Results for Traning Data -----------------------------'
    print 'Percent Accuracy on Training Data = ' + str(accuracyTr) + '%'  
    print 'Percent Accuracy per device on Training Data = ' + str(dev_acc_train) + '%'
    print 'Actual Device Energy on Training Data = ' +  str(actEnergy_train)
    print 'Approx Device Energy on Training Data = ' +  str(appEnergy_train)
    print 'Predicted Device Energy on Training Data = ' +  str(preEnergy_train)
    print 'Computational Expense Classifying Training Data = ' + str(test1Time)  + 's'
    print 'Device Accuracy Approx. vs Actual = ' + str(acTap_train)
    print 'Device Accuracy Pre. vs. Actual = ' + str(acTpre_train)
    print 'Device Accuracy Pre. vs. approx. = ' + str(apTde_train)
    print '------------------------- Results for Test Data -----------------------------'
    print 'Percent Accuracy on Test Data = ' + str(accuracyTs) + '%'    
    print 'Percent Accuracy per device on Test Data = ' + str(dev_acc_test) + '%'
    print 'Actual Device Energy on Test Data = ' +  str(actEnergy_test)
    print 'Approx Device Energy on Test Data = ' +  str(appEnergy_test)
    print 'Predicted Device Energy on Test Data = ' +  str(preEnergy_test)
    print 'Computational Expense Classifying Test Data = ' + str(test2Time)  + 's'
    print 'Device Accuracy Approx. vs Actual = ' + str(acTap_test)
    print 'Device Accuracy Pre. vs. Actual = ' + str(acTpre_test)
    print 'Device Accuracy Pre. vs. approx. = ' + str(apTde_test)
    
    
    

# compute the energy consumption of each device.
    
################################################################
# plot 4 of the devices for illustration
#fig = plt.figure(0)
#lendev = len(device_timer[:,0])
#ax1 = plt.subplot(221)
#plt.plot((device_timer[:,0]-device_timer[0,0])/(device_timer[lendev-1,0]-device_timer[0,0]),device_power[:,0])
#ax1.set_title('Electronics')
#plt.ylabel('Device Power (W)')
#
#ax2 = plt.subplot(222)
#plt.plot((device_timer[:,0]-device_timer[0,0])/(device_timer[lendev-1,0]-device_timer[0,0]),device_power[:,1])
#ax2.set_title('Refrigerator')
##plt.ylabel('Device Power (W)')
#
#ax3 = plt.subplot(223)
#plt.plot((device_timer[:,0]-device_timer[0,0])/(device_timer[lendev-1,0]-device_timer[0,0]),device_power[:,3])
#ax3.set_title('Furnace')
#plt.xlabel('Normalized Time')
#plt.ylabel('Device Power (W)')
#
#ax4 = plt.subplot(224)
#plt.plot((device_timer[:,0]-device_timer[0,0])/(device_timer[lendev-1,0]-device_timer[0,0]),device_power[:,5])
#ax4.set_title('Washer Dryer 2')
#plt.xlabel('Normalized Time')
##plt.ylabel('Device Power (W)')
#
#fig = plt.figure(1)
#plt.plot((device_timer[0:288,0]-device_timer[0,0])/(device_timer[288-1,0]-device_timer[0,0]),device_power[0:288,0])
#
#
#plt.show()
#plt.ylabel('Mains Power Consumption (W)')
#plt.xlabel('time (s)')