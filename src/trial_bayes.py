# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 14:16:33 2015

@author: adeshmu2
"""
import numpy as np
X = np.random.randint(5, size=(6, 100))
y = np.array([1, 2, 2, 4, 5, 6])
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
clf = SVC()


#clf = linear_model.SGDRegressor()
#clf = MultinomialNB()
#MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
clf.fit(X, y)
predicted_labels = clf.predict(X)

diff_labels = predicted_labels - y
accuracy = 1 - float(np.count_nonzero(diff_labels))/float(diff_labels.size)
print 'Percent Accuracy = ' + str(accuracy*100) + '%'  