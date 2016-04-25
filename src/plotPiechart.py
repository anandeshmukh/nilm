# -*- coding: utf-8 -*-
"""
Created on Fri May 08 13:57:40 2015

@author: adeshmu2
"""

from pylab import *
import numpy as np
#from mpltools import style
#from mpltools import layout
import matplotlib.cm as cm
from matplotlib import gridspec

colors = cm.gist_rainbow(np.linspace(0, 1, 9))
#plt.style.use('bmh')

# make a square figure and axes
#figure(1, figsize=(6,6))
#ax = axes([0.1, 0.1, 0.8, 0.8])
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# The slices will be ordered and plotted counter-clockwise.
labels = 'Electronics', 'Refrigerator', 'Dishwasher', 'Furnace','Washer/Dryer 1', 'Washer/Dryer 2', 'Microwave', 'Bathroom GFI','Kitchen Outlets'
#################################################################
# plot the data for Random Forest
fig = plt.figure(0,figsize=(15,6))
#gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2]) 
ax1 = plt.subplot(121)
ax1.set_aspect('equal')
fracs = np.array([1.86852212e+08,2.17513613e+08,3.94863000e+06,1.23842395e+07,1.14932375e+07,1.53197538e+07,6.43228900e+06,4.55543375e+06,1.31138438e+07])
fracs = fracs/np.sum(fracs)
#explode=(0, 0.05, 0, 0)
pie(fracs, autopct='%1.0f%%', colors = colors, shadow=False, startangle=90)
                # The default startangle is 0, which would start
                # the Frogs slice on the x-axis.  With startangle=90,
                # everything is rotated counter-clockwise by 90 degrees,
                # so the plotting starts on the positive y-axis.
title('Actual Device Consumption')
plt.legend(labels,bbox_to_anchor=(1.5, 0.85))
#ax1.legend()
ax2 = plt.subplot(122)
ax2.set_aspect('equal')
fracs = np.array([2.20710188e+08,2.05540643e+08,0.00000000e+00,2.35761301e+06,6.86954365e+06,7.10927986e+06,3.28134646e+05,4.53200405e+06,4.07310771e+05])
fracs = fracs/np.sum(fracs)
pie(fracs, autopct='%1.0f%%', colors = colors, shadow=False, startangle=90)
                # The default startangle is 0, which would start
                # the Frogs slice on the x-axis.  With startangle=90,
                # everything is rotated counter-clockwise by 90 degrees,
                # so the plotting starts on the positive y-axis.
title('Predicted Device Consumption')
show()

#################################################################
# plot the data for Random Forest
fig = plt.figure(1,figsize=(15,6))
#gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2]) 
ax1 = plt.subplot(121)
ax1.set_aspect('equal')
fracs = np.array([1.86852212e+08,2.17513613e+08,3.94863000e+06,1.23842395e+07,1.14932375e+07,1.53197538e+07,6.43228900e+06,4.55543375e+06,1.31138438e+07])
fracs = fracs/np.sum(fracs)
#explode=(0, 0.05, 0, 0)
pie(fracs, autopct='%1.0f%%', colors = colors, shadow=False, startangle=90)
                # The default startangle is 0, which would start
                # the Frogs slice on the x-axis.  With startangle=90,
                # everything is rotated counter-clockwise by 90 degrees,
                # so the plotting starts on the positive y-axis.
title('Actual Device Consumption')
plt.legend(labels,bbox_to_anchor=(1.5, 0.85))
#ax1.legend()
ax2 = plt.subplot(122)
ax2.set_aspect('equal')
fracs = np.array([9161.66666667,2507.33333333,0.,544.16666667,11.66666667,11.66666667,35.,0.,226.5])
fracs = fracs/np.sum(fracs)
pie(fracs, autopct='%1.0f%%', colors = colors, shadow=False, startangle=90)
                # The default startangle is 0, which would start
                # the Frogs slice on the x-axis.  With startangle=90,
                # everything is rotated counter-clockwise by 90 degrees,
                # so the plotting starts on the positive y-axis.
title('Predicted Device Consumption')
show()