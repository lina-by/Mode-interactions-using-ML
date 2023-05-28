import numpy as np
import pickle
import No_ML as nml

with open(r"svm.pkl", 'rb') as file:
    default = pickle.load(file)

# returns a dictionary, the keys correspond to the interaction and the values are the interaction score for each interaction.
def interaction(scalogram, model=default, scaled=True, frequencies=None, f0=None, fend=None):
    if not scaled:
        scalogram = scale(scalogram, frequencies, f0, fend)
    n=np.shape(scalogram)[1]
    proba=model.predict_proba(scalogram.T)
    score={}
    score[2]=np.sum(proba[:,1])/n
    score[3]=np.sum(proba[:,2])/n
    score[4]=0
    score[5]=np.sum(proba[:,3])/n
	#for i in range(2,6)
	#	score[i]=np.sum(proba[:,i])/n
    return score

# Creates the SVM with the wished threshold.
def model(threshold=2, chosen_kernel='linear'):
    classes=[0,2,3,4,5]
    X=np.genfromtxt(r"X.csv", delimiter=",", dtype=float)
    y=np.genfromtxt(r"y.csv", delimiter=",", dtype=float)
    y=y[:,0]*(y[:,1]>threshold)
    
	# getting rid of class imbalance
	count=[]
	for i in classes:
	    count.append(np.sum(y==i))
    ClassSetSize = min(count) # nb of samples for each class
    if ClassSetSize<100:
	    print("the chosen threshold is such that some classes do not have sufficient data points")
	    return None
    
	newX=[] 
    newY=[]
	counters=[0]*6
    import random as rd    
    indices=[np.sort(rd.sample(range(count[i]), ClassSetSize)) for i in range(5)] #randomly choosing which vectors to use to train the machine
    
    for x in range(len(y)):
	    i=y[x] # i in an element of classes, i.e. 0,2,3,4,5
	    j=0 # which correspond to positions j=0,1,2,3,4 in lists count or indices
	    if i>0:
	    	j=i-1
	    if counters[j] == indices[j][0]:
			indices[j].pop(0)
			newX.append(X[x])
			newY.append(i)
	    counters[j]+=1
    newX=np.array(newX)
    newY=np.array(newY)
	    
    # training the model
    from sklearn.svm import SVC
    svm = SVC(C = 100, gamma='auto', kernel=chosen_kernel, probability=True)
    svm.fit(newX,newY)
    return svm

# Prints the accuracy on training and testing sets.
def Cross_validate(model, X, y, k=5):
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=k, shuffle=True)
	tot_correct_fraction_test = 0
	tot_correct_fraction_train = 0
	for train_index, test_index in kf.split(X,y):
	# creating training and testing dataset
		X_train = X[train_index]
		y_train = y[train_index]
		X_test = X[test_index]
		y_test = y[test_index]
	# fit the training dataset
		model.fit(X_train,y_train)

		#use model.predict() to predict the output for the test data set
		y_test_model = model.predict(X_test)
	
		#loop through to compare the test data output to what it should be and obtain the fraction of correct classifications
		nTot = len(y_test) 
		nMatch = 0 
		for i in range(len(y_test)):
			if y_test[i] == y_test_model[i]:
				nMatch += 1

		correct_fraction_test = nMatch / nTot

		#do the same prediction and performance assessment performance with the training data
		y_train_model = model.predict(X_train)

		nTot = len(y_train) 
		nMatch = 0
		for i in range(len(y_train)):
			if y_train[i] == y_train_model[i]:
				nMatch += 1

		correct_fraction_train = nMatch / nTot

		#add on to the totals
		tot_correct_fraction_test += correct_fraction_test
		tot_correct_fraction_train += correct_fraction_train
	print('Correct percentage for test data: ', 100*tot_correct_fraction_test/5)
	print('Correct percentage for training data: ', 100*tot_correct_fraction_train/5)