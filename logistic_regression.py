import extract as extract
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def build_train(filename):
	extracted = extract.extract(filename)
	train_x = []
	train_y = [] 
	for example in extracted:
		similarity = cosine_similarity(example["reference"],example["candidate"])
		train_x.append([float(example["bleu"]),float(similarity)])
		train_y.append(0 if example["label"] == "H" else 1)
	return np.array(train_x), np.array(train_y)

def build_test(filename):
	extracted = extract.extract(filename)
	test_x = []
	test_y = []
	for example in extracted:
		similarity = cosine_similarity(example["reference"],example["candidate"])
		test_x.append([float(example["bleu"]),float(similarity)])
		test_y.append(0 if example["label"] == "H" else 1)
	return np.array(test_x), np.array(test_y)
	
def cosine_similarity(x,y):
	# tokenization 
	X_list = word_tokenize(x)  
	Y_list = word_tokenize(y)   
	l1 =[];l2 =[]  
	X_set = set(X_list)
	Y_set = set(Y_list)
	# form a set containing keywords of both strings  
	rvector = X_set.union(Y_set)  
	for w in rvector: 
	    if w in X_set: l1.append(1) # create a vector 
	    else: l1.append(0) 
	    if w in Y_set: l2.append(1) 
	    else: l2.append(0) 
	c = 0
	# cosine formula  
	for i in range(len(rvector)): 
	        c+= l1[i]*l2[i] 
	cosine = c / float((sum(l1)*sum(l2))**0.5) 
	return cosine


'''
Logistic Regression:
	- Input: (1) Bleu Score, (2) Cosine Similarity
'''
def logistic_regressor_train(clf):
	train_data_x, train_data_y = build_train("train.txt")
	clf.fit(train_data_x, train_data_y)
	return clf

def predict(clf):
	test_data_x, test_data_y = build_test("test.txt")
	predicted = []
	for sample in test_data_x:
		proc_sample = np.array([list(sample)])
		print(proc_sample)
		pred = clf.predict(proc_sample)[0]
		predicted.append(pred)
	return predicted, test_data_y

def accuracy(ground_truth, prediction):
	total = len(ground_truth)
	correct = 0
	for i in range(total):
		if ground_truth[i] == prediction[i]:
			correct += 1
	return float(correct)/total

def f1score(ground_truth, prediction, average='macro'):
	return f1_score(ground_truth, prediction)

def classify():
	clf = LogisticRegression()
	clf = logistic_regressor_train(clf)
	pred, ground_truth = predict(clf)
	print(pred)
	acc = accuracy(ground_truth,pred)
	print("The % Accuracy is: " + str(float(acc*100)) + "%")
	print("The F1 Score computed using Sklearn is: " + str(f1score(ground_truth, pred)))
	X, Y = build_train("train.txt")
	plot(X, Y, clf)


def plot(X,Y,clf):
	# Plot the decision boundary. For that, we will assign a color to each
	# point in the mesh [x_min, x_max]x[y_min, y_max].
	font = {'weight' : 'bold',
	'size'   : 14}
	plt.rc('font', **font)
	ax = plt.gca()
	x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	h = .02  # step size in the mesh
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.figure(1, figsize=(4, 3))
	plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, shading="auto")
	# Plot also the training points
	plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
	plt.xlabel('Bleu Score')
	plt.ylabel('Cosine Similarity')
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	ax.set_title("Logistic Regression Decision Boundary")
	plt.xticks(())
	plt.yticks(())
	plt.show()

classify()


