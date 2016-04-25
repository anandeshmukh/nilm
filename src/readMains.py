# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:32:49 2015

@author: adeshmu2

"""
import numpy as np
import matplotlib.pyplot as plt

datapath = 'Data/house_3/'
mains_timer = np.zeros(shape=(1427284,2))
mains_power = np.zeros(shape=(1427284,2))
total_mains_power = np.zeros(shape=(1427284))
###
#
for mains in range(0,1):
    channel = mains + 1
    filename = 'channel_'+ str(channel) +'.dat'
    filepath = datapath + filename
    xtemp,ytemp = np.loadtxt(filepath,unpack=True)
    mains_timer[:,mains] = xtemp
    mains_power[:,mains] = ytemp
    total_mains_power += ytemp
    #plt.plot(mains_timer[:,mains],mains_power[:,device])
##
plt.figure(0)
plt.plot(mains_timer[:,mains],total_mains_power)
plt.show()
#plt.ylabel('Mains Power Consumption (W)')
#plt.xlabel('time (s)')
#plt.legend()
