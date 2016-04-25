# -*- coding: utf-8 -*-
"""
Created on Thu May 07 16:43:11 2015

@author: Danny
"""

import numpy as np
from MLL import bitfield

def bitconv(x,nd):
    bitvector = np.zeros(shape=(nd)) 
    remarray = bitfield(x)
    bitvector[nd-len(remarray):nd] = remarray
    return bitvector
                  
def bitconvarray(xarray,nd):
    ni = len(xarray)
    binarylabels = np.zeros(shape=(ni,nd)) 
    for i in range(0,ni):
        binarylabels[i,:] = bitconv(xarray[i],nd)
    return binarylabels