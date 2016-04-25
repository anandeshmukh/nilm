# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:53:34 2015

@author: adeshmu2
"""
import numpy as np
import time
from bitconv import bitconvarray

# extract features, create instances and create labels from the test data.
def createInstances(total_device_power, device_timer, device_power,weather_data,classify,timewindow):
    timestep = device_timer[2,1] - device_timer[1,1]
    numdata = len(total_device_power)
    numdevices = len(device_power[0])
    idxstep = int(timewindow/timestep)
    numinstances = int(numdata/idxstep)
    binarylabels = np.zeros(shape=(numinstances,numdevices),dtype=np.int)
    
    # create 5 minute snippets from the given data.
    stridx = 0
    endidx = idxstep - 1
    snippets = np.zeros(shape=(numinstances,idxstep))
    snippets_timer = np.zeros(shape=(numinstances,idxstep))
    snippets_tstamp = np.zeros(shape=(numinstances))
    snippets_temp = np.zeros(shape=(numinstances))
    snippets_devices = np.zeros(shape=(idxstep,numdevices))
    labels = np.zeros(shape=(numinstances))
    
    for instance in range(0,numinstances):
        snippets[instance,:] = total_device_power[stridx:endidx+1]
        snippets_timer[instance,:] = device_timer[stridx:endidx+1,0]
        snippets_tstamp[instance] = device_timer[endidx+1,0]
        snippets_temp[instance] = weather_data[endidx+1]
        for device in range(0,numdevices):
            snippets_devices[:,device] = device_power[stridx:endidx+1,device]
        # get the correct labels
        labels[instance],binarylabels[instance,:] = extractLabel(snippets_devices,numdevices)
        stridx = endidx + 1
        endidx = endidx + idxstep
    
    if classify == 1:
        instances = extractFeaturesBayes(snippets,snippets_tstamp,snippets_temp,numinstances,weather_data,snippets_timer)    
    elif classify == 2 or classify == 3 or classify == 4 or classify == 5 or classify == 6:
        instances = extractFeaturesRegression(snippets,snippets_tstamp,snippets_temp,numinstances,weather_data,snippets_timer)    
        
    return instances, labels, binarylabels
    
def extractFeaturesBayes(snippets,snippets_tstamp,snippets_temp,numinstances,weather_data,snippets_timer):
    instances = np.zeros(shape=(numinstances,7))
    
    for instance in range(0,numinstances):
        # feature 0 - average power instance        
        avg_instance = np.average(snippets[instance,:])        
        instances[instance,0] = avgRanking(avg_instance)
                
        # feature 1 - std deviation of power        
        std_instance = np.std(snippets[instance,:])        
        instances[instance,1] = stdRanking(std_instance)
        
        # feature 2 - local hour of the day fraction
        local_time_sec = time.localtime(snippets_tstamp[instance])
        time_hour = float(local_time_sec.tm_hour) + float(local_time_sec.tm_min)/60
        instances[instance,2] = todRanking(time_hour)
        
        # feature 3 - local average temperature during the 5 minute window      
        instances[instance,3] = weatherRanking(snippets_temp[instance])
 
        # feature 4 - Maximum power reading
        instances[instance, 4] = maxPowerRanking(max(snippets[instance,:]))

        # feature 5 - The energy (integral of power) reading
        instances[instance, 5] = energyRanking(np.trapz(snippets[instance,:],snippets_timer[instance,:]))
     
        # feature 6 - Day of the week 
        instances[instance,6] = local_time_sec.tm_wday
        
    return instances  

def extractFeaturesRegression(snippets,snippets_tstamp,snippets_temp,numinstances,weather_data,snippets_timer):
    instances = np.zeros(shape=(numinstances,7))
    
    for instance in range(0,numinstances):
        # feature 0 - average power instance        
        avg_instance = np.average(snippets[instance,:])        
        instances[instance,0] = avg_instance
                
        # feature 1 - std deviation of power        
        std_instance = np.std(snippets[instance,:])        
        instances[instance,1] = std_instance
        
        # feature 2 - local hour of the day fraction
        local_time_sec = time.localtime(snippets_tstamp[instance])
        time_hour = float(local_time_sec.tm_hour) + float(local_time_sec.tm_min)/60
        instances[instance,2] = todRanking(time_hour)
        
        # feature 3 - local average temperature during the 5 minute window      
        instances[instance,3] = snippets_temp[instance]
      
        # feature 4 - Maximum power reading
        instances[instance, 4] = max(snippets[instance,:])
      
        # feature 5 - The energy (integral of power) reading
        instances[instance, 5] = np.trapz(snippets[instance,:],snippets_timer[instance,:])

        # feature 6 - Day of the week 
        instances[instance,6] = local_time_sec.tm_wday
   
    return instances

def extractLabel(snippets_devices,numdevices):
    label = np.zeros(shape=(numdevices),dtype=np.int)
    #multiplier = np.array([512]    
    #binarylabel = np.zeros()
    for device in range(0,numdevices):
        label[device] = deviceStatus(snippets_devices[:,device])
    
    binarylabel = label
    labelstr = ''.join(str(x) for x in list(label))
    labelint = int(labelstr,2)  
    return labelint, binarylabel
    
def deviceStatus(device_snippet):
    if (np.max(device_snippet) >= 100):
        status = 1
    else:
        status = 0
    return status
         
def avgRanking(x):
    a = 40 # Number of bins
    for i in range (1,a+2):        
        lb = 4000/a*(i-1)
        ub = 4000/a*(i)
        if (x >= lb and x <= ub):
            rank = i
        if(x > 4000):
            rank = 0
    return rank  
        
def stdRanking(x):        
    a = 10 # Number of bins
    for i in range (1,a+2):        
        lb = 1500/a*(i-1)
        ub = 1500/a*(i)
        if (x >= lb and x <= ub):
            rank = i
        if(x > 1500):
            rank = 0
    return rank
        
def todRanking(x):
    if (x >= 6 and x <= 10):
        rank = 1
    elif (x > 10 and x <= 15):
        rank = 2
    elif (x > 15 and x <= 22):
        rank = 3
    else:
        rank = 4 
    return rank        
        
def weatherRanking(x):      
    a = 10 # Number of bins
    for i in range (1,a+2):        
        lb = 100/a*(i-1)
        ub = 100/a*(i)
        if (x >= lb and x <= ub):
            rank = i
        if(x > 100):
            rank = 0
        if(x < 0):
            rank = a+2 
    return rank     


def maxPowerRanking(x):
    a = 10 # Number of bins
    for i in range (1,a+2):        
        lb = 8000/a*(i-1)
        ub = 8000/a*(i)
        if (x >= lb and x <= ub):
            rank = i
        if(x > 8000):
            rank = 0
        if(x < 0):
            rank = a+2 
    return rank            

def energyRanking(x):
    a = 10 # Number of bins
    for i in range (1,a+2):        
        lb = 100000/a*(i-1)
        ub = 100000/a*(i)
        if (x >= lb and x <= ub):
            rank = i
        if(x > 100000):
            rank = 0
    return rank

def deviceErrors(clf,nd,instances,actual_labels,actual_labels_binary):
    predicted_labels_temp = clf.predict(instances)
    predicted_labels = predicted_labels_temp.astype(int)
    predicted_labels_binary = bitconvarray(predicted_labels,nd)
    diff_labels = predicted_labels - actual_labels
    diff_labels_binary = predicted_labels_binary - actual_labels_binary
    diff_labels_binary_abs =  np.abs(diff_labels_binary)
    error_sums = np.sum(diff_labels_binary_abs,axis=0)
    error_sums = error_sums*100/len(actual_labels)
    dev_accuracy = 100 - error_sums
    accuracy = 1 - float(np.count_nonzero(diff_labels))/float(diff_labels.size)
    accuracy = accuracy*100
    return accuracy,dev_accuracy,predicted_labels_binary
    