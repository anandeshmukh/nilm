# -*- coding: utf-8 -*-
"""
Created on Thu May 07 18:31:41 2015

@author: Danny
"""
import numpy as np

def actDevEnergy(device_power,device_timer,nd):
    actual_device_energy = np.zeros(shape=(nd))
    for i in range(0,nd):
        actual_device_energy[i] = np.trapz(device_power[:,i],device_timer[:,i])
    return actual_device_energy  
    

def appDevEnergy(labels_binary,agg_energy,nd):
    approx_device_energy = np.zeros(shape=(nd))
    weights = np.sum(labels_binary,axis=1)
    for j in range(0,len(weights)):
        if not weights[j]:
            weights[j] = 1    
    agg_energy = agg_energy/weights.astype(float)    
    for i in range(0,nd):
        approx_device_energy[i] = np.dot(labels_binary[:,i],agg_energy)
    return approx_device_energy  

def energyComp(AcDE, ApDE, PreDE):
    acTap = (abs(ApDE - AcDE)/AcDE)*100
    acTpre = (abs(PreDE - AcDE)/AcDE)*100
    apTde = (abs(PreDE - ApDE)/ApDE)*100
    return acTap, acTpre, apTde