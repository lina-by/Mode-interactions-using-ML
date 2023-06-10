from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

#i corresponds to the class {0,2,3,4,5}, j is the position if those classes in lists
def indice(i):
    j=0
    if i>0:
	    j=i-1
    return j

# Prints the accuracy on training and testing sets.
def Cross_validate(model, X, y, k=5):
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=k, shuffle=True)

	TP=[0]*5 # one per class
	FP=[0]*5
	FN=[0]*5
	TN=[0]*5
	accuracy=0

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
		for x in range(len(y_test)):
			i=int(y_test[x])
			ipred=int(y_test_model[x])
			#i corresponds to the class {0,2,3,4,5}, j is the position if those classes in lists
			j, jpred = indice(i), indice(ipred)
			
			if i == ipred:
				accuracy+=1
				TP[j]+=1
				for k in range(5):
					if k!=j:
						TN[k]+=1
			
			else:
				FN[j]+=1
				FP[jpred]+=1
				for k in range(5):
					if k!=j and k!=jpred:
						TN[k]+=1

	total=TN[0]+TP[0]+FN[0]+FP[0]
	accuracy=accuracy/total

	dic = {'accuracy': accuracy}
	for i in [0,2,3,4,5]:
		j=indice(i)
		dic2 = {'TN rate': TN[j]/total,
	  			'FN rate': FN[j]/total,
	  			'TP rate': TP[j]/total,
	  			'FP rate': FP[j]/total,
	  			'precision': TP[j]/(TP[j]+FP[j]),
	  			'sensitivity': TP[j]/(TP[j]+FN[j]),
	  			'specificity': TN[j]/(TN[j]+FP[j]),
	  			}
		dic[i]=dic2
	return dic


# Prints the accuracy on training and testing sets.
def calculate_bias_variance(model, X, y, k=5):
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=k, shuffle=True)
	bias_scores = []
	variance_scores = []
	for train_index, test_index in kf.split(X,y):
		# creating training and testing dataset
		X_train = X[train_index]
		y_train = y[train_index]
		X_test = X[test_index]
		y_test = y[test_index]
		# fit the training dataset
		model.fit(X_train,y_train)
		#use model.predict() to predict the output for the test data set
		y_pred = model.predict(X_test)
		# Calculate accuracy score as the performance metric
		accuracy = accuracy_score(y_test, y_pred)
    	# Calculate bias and variance
		bias = (1 - accuracy)
		variance = np.var(y_pred != y_test)

        # Append bias and variance scores to the lists
		bias_scores.append(bias)
		variance_scores.append(variance)

    # Calculate average bias and variance
	bias = np.mean(bias_scores)
	variance = np.mean(variance_scores)
	return bias, variance

def ROC_probability(model, X, y, probas, k=5):
	dictionaries=[]
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=k, shuffle=True)
	n=len(probas)
	TP=[[0]*5 for _ in range(n)]
	FP=[[0]*5 for _ in range(n)]
	FN=[[0]*5 for _ in range(n)]
	TN=[[0]*5 for _ in range(n)]

	for train_index, test_index in kf.split(X,y):
	# creating training and testing dataset
		X_train = X[train_index]
		y_train = y[train_index]
		X_test = X[test_index]
		y_test = y[test_index]
	# fit the training dataset
		model.fit(X_train,y_train)
		y_test_model = model.predict_proba(X_test)
	#use model.predict() to predict the output for the test data set
		for l in range(n):
			proba=probas[l]
			y_test_model2=y_test_model>proba
			
		#loop through to compare the test data output to what it should be and obtain the fraction of correct classifications			
			for x in range(len(y_test)):
				i=int(y_test[x]) #i corresponds to the class {0,2,3,4,5}, j is the position if those classes in lists
				j=indice(i)
				pred=y_test_model2[x,:]
							
				if pred[j] == 1:
					TP[l][j]+=1
					for k in range(5):
						if k!=j:
							if pred[k]==1:
								FP[l][k]+=1
							else:
								TN[l][k]+=1
				
				else:
					FN[l][j]+=1
					for k in range(5):
						if k!=j:
							if pred[k]==1:
								FP[l][k]+=1
							else:
								TN[l][k]+=1

	total=TN[l][0]+TP[l][0]+FN[l][0]+FP[l][0]
	for l in range(n):
		dic = {}
		for i in [0,2,3,4,5]:
			j=indice(i)
			dic2 = {'TN rate': TN[l][j]/total,
					'FN rate': FN[l][j]/total,
					'TP rate': TP[l][j]/total,
					'FP rate': FP[l][j]/total
					}
			dic[i]=dic2
		dictionaries.append(dic)
	return dictionaries

def method_compare(Sets, ax, columns):
    n=len(Sets)
    data=np.zeros((4,n))
    for i in range(n):
        set=Sets[i]
        l=[set[2],set[3],set[4],set[5]]
        data[:,i]=l

    rows = ["1:2","1:3","1:4","1:5"]
    # Get some pastel shades for the colors
    colors = plt.cm.BuPu(np.linspace(0.5, 0.1, len(rows)))
    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        ax.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
        y_offset = y_offset + data[row]
        cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])
    # Reverse colors and text labels to display the last value at the top.
    #cell_text.reverse()
    #data.reverse()
    #colors = colors[::-1]

    # Add a table at the bottom of the axes
    the_table = ax.table(cellText=data,
                        rowLabels=rows,
                        rowColours=colors,
                        colLabels=columns,
                        loc='bottom')

    # Adjust layout to make room for the table:
    #ax.subplots_adjust(left=0.2, bottom=0.2)

    ax.set_ylabel("Score")
    ax.set_xticks([])
    ax.set_title('Comparison between ML and not ML methods')