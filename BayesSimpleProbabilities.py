# Mushroom Data Set
# https://archive.ics.uci.edu/ml/datasets/Mushroom
# Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
# Irvine, CA: University of California, School of Information and Computer Science.

# Python version: 3.5.4
# Numpy version: 1.14.0
# SciKit-Learn version : 0.19.1

# @desc Use Naiv Bayes Classifier on Mushroom Data Set Simple Probabilities
# This is a first implementation of this sort
# @author Cramer Grimes cramergrimes@gmail.com

import numpy as np
from sklearn.model_selection import train_test_split

data =  np.genfromtxt('agaricus-lepiota.data',delimiter=',',dtype='str')

y = data[:,0]
X = data[:,1:]


Cn=len(np.unique(y))

n,d = X.shape

print ("initial samples: {}".format(n))
print ("number of features: {}".format(d))
print ("number of class labels: {}".format(Cn))
print ()

print ("Class Labels are: {}".format(np.unique(y)))
print ()

print ("Take a look at unique outcomes per feature")
for i in range(0,d):
	print ("{}th: {}".format(i,np.unique(X[:,i])))

X = np.delete(X,(10,15),axis=1)

print ()
print ("Remove 10th feature because it has some missing data")
print ("Remove 15th feature because it is always 'p'")

n,d = X.shape

# dictionary master list of unique features
featureDict = {}
for i in range(0,d):
	featureDict[i]= np.unique(X[:,i])

print ()
print ("After removing the two features")
print ("number of features: {}".format(d))

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)
n_train = len(X_train)
n_test = len(X_test)

print ()
print ("number of training samples: {}".format(n_train))
print ("number of test samples: {}".format(n_test))

# Isolate the training set based on clasification label
X_train_e = X_train[y_train=='e']
X_train_p = X_train[y_train=='p']

# capture number of each class label in training set
n_train_e = len(X_train_e)
n_train_p = len(X_train_p)

# two dictionaries to capture likelihoods (features given class labels)
featureGivenEd = {}
featureGivenPo = {}

for i in range(0,d):
	# start with edible first
	# gather unique feautre outcomes and their counts
	unique, counts = np.unique(X_train_e[:,i], return_counts=True)
	# zip things up into a dictionary
	countDict = dict(zip(unique,counts))
	# temporary dictionary to hold probabilities
	prob_e = {}
	for cha in featureDict[i]:
		if cha in countDict:
			# pacture simple probabilities
			prob_e[cha]=float(countDict[cha]/n_train_e)
		else:
			prob_e[cha]=float(0)
	# add the temporary dictionary to the main one
	featureGivenEd[i]=prob_e

	# do the same all again for poisonous samples
	unique, counts = np.unique(X_train_p[:,i], return_counts=True)
	countDict = dict(zip(unique,counts))
	prob_p = {}
	for cha in featureDict[i]:
		if cha in countDict:
			prob_p[cha]=float(countDict[cha]/n_train_p)
		else:
			prob_p[cha]= float(0)
	featureGivenPo[i]=prob_p

# let's not forget about the prior probabilities
priorEd =float(n_train_e/n_train)
priorPo =float(n_train_p/n_train)

# matrix to hold the posterior probabilities
class_given_data_mat = np.zeros((n_test,Cn), dtype=float)


for i in range(0,n_test):
	# eProb is a running total of an individual posterior probability
	eProb = priorEd
	# pProb is a running total of an individual posterior probability
	pProb = priorPo
	for j in range(0,d):
		# continually multiply likelihood probabilities to our running totals
		eProb = eProb * featureGivenEd[j][X_test[i][j]]
		pProb = pProb * featureGivenPo[j][X_test[i][j]]
	# update the matrix holding our posterior probabilities
	class_given_data_mat[i][0]=eProb
	class_given_data_mat[i][1]=pProb

# need to make predictions in therms of 'e' and 'p'
y_test_pred = np.zeros((n_test), dtype='str')
for i in range(0,n_test):
	if np.argmax(class_given_data_mat[i]) == 0:
		y_test_pred[i]='e'
	else:
		y_test_pred[i]='p'

print ()
print ("Probability of correct prediction")
print ((y_test_pred==y_test).sum()/n_test)