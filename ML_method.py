import numpy as np
import pickle
import No_ML as nml

with open(r"balanced-svm-thr2.pkl", 'rb') as file:
    default = pickle.load(file)

# returns a dictionary, the keys correspond to the interaction and the values are the interaction score for each interaction.
def interaction(scalogram, model=default, scaled=True, frequencies=None, f0=None, fend=None):
    if not scaled:
        scalogram = scale(scalogram, frequencies, f0, fend)
    n=np.shape(scalogram)[1]
    proba=model.predict_proba(scalogram.T)
    score={}
    for i in range(2,6):
          score[i]=np.sum(proba[:,i-1])/n
    return score


# Creates the SVM with the wished threshold.
def model(threshold=2, Xy='default', chosen_kernel='linear', chosen_C=100, chosen_gamma='auto'):
    classes=[0,2,3,4,5]
    if Xy =='default':
         X=np.genfromtxt(r"X.csv", delimiter=",", dtype=float)
         y=np.genfromtxt(r"y.csv", delimiter=",", dtype=float)
         X=X.reshape((len(X)//120,120))
         y=y.reshape((len(y)//2,2))
    else:
         X=Xy[0]
         y=Xy[1]

    y=y[:,0]*(y[:,1]>threshold)    
	# getting rid of class imbalance
    count = [np.sum(y==i) for i in classes]
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
        i=int(y[x]) # i in an element of classes, i.e. 0,2,3,4,5
        j=0 # which correspond to positions j=0,1,2,3,4 in lists count or indices
        if i>0:
             j=i-1
        
        if len(indices[j]) == 0:
            length = [len(x) for x in indices]
            maxlen=max(length)
            if maxlen==0:
                break
            continue
        if counters[j] == indices[j][0]:
            indices[j]=indices[j][1:]
            newX.append(X[x])
            newY.append(i)
        counters[j]+=1
    newX=np.array(newX)
    newY=np.array(newY)
	    
    # training the model
    from sklearn.svm import SVC
    svm = SVC(C = chosen_C, gamma=chosen_gamma, kernel=chosen_kernel, probability=True)
    svm.fit(newX,newY)
    return newX, newY, svm

# same function as model but ignores the issue of class imbalace
def model2(threshold=2, Xy='default', chosen_kernel='linear', chosen_C=100, chosen_gamma=0.1):
    classes=[0,2,3,4,5]
    if Xy =='default':
         X=np.genfromtxt(r"X.csv", delimiter=",", dtype=float)
         y=np.genfromtxt(r"y.csv", delimiter=",", dtype=float)
         X=X.reshape((len(X)//120,120))
         y=y.reshape((len(y)//2,2))
    else:
         X=Xy[0]
         y=Xy[1]

    y=y[:,0]*(y[:,1]>threshold)    
     # training the model
    from sklearn.svm import SVC
    svm = SVC(C = chosen_C, gamma=chosen_gamma, kernel=chosen_kernel, probability=True)
    svm.fit(X,y)
    return svm

