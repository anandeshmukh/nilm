# -*- coding: utf-8 -*-
"""
Created on Fri May 08 18:49:05 2015

@author: adeshmu2
"""

import numpy as np

# the array of values for 5-fold cross validation
naive_accs = np.array([41.7833333333,38.675,41.725,34.775,43.2])
forest_accs = np.array([75.7916666667,70.25,74.25,77.9,73.95])
naive_dev_accs = np.zeros(shape=(5,9))
forest_dev_accs = np.zeros(shape=(5,9))

naive_dev_accs[0,:] = np.array([ 59.28333333,73.825,99.18333333,96.5,99.68333333,98.0,99.45833333,98.425,89.34166667])
naive_dev_accs[1,:] = np.array([ 76.75,87.275,98.325,94.35,99.6,98.075,99.,95.325,92.75])
naive_dev_accs[2,:] = np.array([ 82.15,89.15,98.175,94.2,99.65,96.8,98.75,97.375,94.45])
naive_dev_accs[3,:] = np.array([ 84.275,87.15,99.1,98.4,99.9,98.,99.275,98.6,97.75 ])
naive_dev_accs[4,:] = np.array([ 58.325,78.15,98.825,98.375,98.825,97.1,98.8,97.5,97.275])


forest_dev_accs[0,:] = np.array([ 87.88333333,88.09166667,99.18333333,98.15,99.90833333,97.90833333,99.65833333,98.975,89.475])
forest_dev_accs[1,:] = np.array([ 76.75,87.275,98.325,94.35,99.6,98.075,99.,95.325,92.75 ])
forest_dev_accs[2,:] = np.array([ 82.15,89.15,98.175,94.2,99.65,96.8,98.75,97.375,94.45 ])
forest_dev_accs[3,:] = np.array([ 84.275,87.15,99.1,98.4,99.9,98.,99.275,98.6,97.75 ])
forest_dev_accs[4,:] = np.array([ 79.2,87.775,99.05,99.275,98.075,96.425,99.325,98.7,98.075])

naivestd = np.std(naive_accs)
foreststd = np.std(forest_accs)
naiveavg = np.average(naive_accs)
forestavg = np.average(forest_accs)

naive_dev_avg = np.average(naive_dev_accs,axis=0)
forest_dev_avg = np.average(forest_dev_accs,axis=0)

print 'Naive average = ' + str(naiveavg)
print 'Naive std = ' + str(naivestd)
print 'Forest average = ' + str(forestavg)
print 'Forest std = ' + str(foreststd)
print 'Naive device average = ' + str(naive_dev_avg)
print 'Forest device average = ' + str(forest_dev_avg)