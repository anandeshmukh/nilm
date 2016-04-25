# -*- coding: utf-8 -*-
"""
Created on Thu May 07 09:09:46 2015

@author: Danny
"""

import numpy as np
from sklearn import linear_model
n_samples, n_features = 6, 5
np.random.seed(0)
y = np.array([1, 2, 3, 4, 5, 6])
X = np.random.randn(n_samples, n_features)
clf = linear_model.SGDRegressor()
clf.fit(X, y)
print(clf.predict(X[4]))