import csv                                                                    #Library for .csv files
import numpy as np                                                            #Library for numerical arrays
from sklearn.naive_bayes import GaussianNB                                    #Library for machine learning -> Gaussian Naive Bayes Algorithm

#Read function used to read data from .csv files
def read_data(fn):                                                            #fn -> File name of file to be read
    file_name = fn
    raw_data = open(file_name ,'rt')                                          #Open file in read mode
    data = csv.reader(raw_data ,delimiter = ',' ,quoting = csv.QUOTE_NONE )   #Object read from csv file  
    X = list ( data )                                                         #Formatted list from object
    return X

#Format the data suitably for the Algorithm
def format_data(X):
    str_ind =[1,2,3,41]                                                       #List of column indices with strings as values
    all_dicts = []                                                            #List of dictionaries used for data formatting
    for j in str_ind:                                                         #For each index j in list of indices
        col_list =[]
        for i in range(len(X)):                                               #Append all values in that column
            col_list.append(X[i][j]) 
        uniq_list = set(col_list)                                             #Form a set of distinct values only
        uniq_list = sorted(uniq_list)                                         #Sort the set
        k=0
        dict1 = {}
        for ele in uniq_list:                                                 #For each string in the set
            dict1[ele] = k                                                    #Assign string to index k in the dictionary
            k+=1
        all_dicts.append(dict1)                                               #Add current dictionary
        for i in range(len(X)):                                               #For all values in that column
            X[i][j] = dict1.get(X[i][j])                                      #Substitute the index in place of string
    return [X, all_dicts]                                                     #Formatted data and dictionaries used

#Partition the complete data into inputs and corresponding outputs(labels)
def split_data(X):                                                            
    train_data, train_target = [], []
    for i in range(1,len(X)):                                                 #For each row from 2nd row (1st row contains titles of fields)
        curr_row = []
        for j in range(len(X[i])-1):                                          #For each value in that row except the last -> inputs
            curr_row.append(float(X[i][j]))                                   #Type cast from string to float
        train_data.append(curr_row)                                           #Append row
        train_target.append(float(X[i][len(X[i])-1]))                         #Append last element in the row -> output
    train_data = np.array(train_data)
    train_target = np.array(train_target)                                     #Format into standard arrays
    return [train_data, train_target]

#Return class of data based on the integer predicted(derived from the dictionary)
def class_data(yi): 
  if(int(yi) == 0):
    y1 = 'anamoly'
  elif(int(yi) == 1):
    y1 = 'class'
  else:
    y1 = 'normal'
  return y1                                                                   #Return corresponding string

#Read training data
X = read_data('train1.csv')                                                   #Call read_data() with training data file
X, all_dicts = format_data(X)                                                 #Format data
tr_data, tr_target = split_data(X)                                            #Split data into standard inputs and outputs

#Display dictionaries used
print("Dictionaries used for data conversion:\n")
for d1 in all_dicts:
    print(d1)
    
#Fit the data
gnb = GaussianNB()                                                            #Declare an object of Gaussian Naive Bayes Algorithm
clf = gnb.fit(tr_data, tr_target)                                             #Fit the corresponding inputs and outputs and assign this object to the classifier

#Read testing data
X = read_data('test.csv')                                                     #Call read_data() with testing data file                                                     
X, all_dicts = format_data(X)                                                 #Format data
te_data, te_target = split_data(X)                                            #Split data into standard inputs and outputs

#Predict the test targets
tr_pred = clf.predict(tr_data)

print("\n---------------\nTraining Dataset\n---------------\nPredicted_Class\tActual_Class\n---------------\t-------------")
c1, ct1 = 0, 0
for i in range(20):                                                           #Display sample training data predictions
    yi, di = tr_pred[i], tr_target[i]
    print("{}\t\t{}".format(y1, d1))
    
        
for yi, di in zip(tr_pred, tr_target):                                        #For each prediction and expected output
	if(int(yi) == int(di)):
		c1 += 1                                                                   #Count the correct predictions
	ct1 += 1                                                                    #Counter for total number of predictions

#Predict the test targets
te_pred = clf.predict(te_data)

print("\n---------------\nTesting Dataset\n---------------\nPredicted_Class\tActual_Class\n---------------\t-------------")
c, ct = 0, 0
for i in range(20):                                                           #Display sample testing data predictions
    yi, di = te_pred[i], te_target[i]
    y1, d1 = class_data(yi), class_data(di)
    print("{}\t\t{}".format(y1, d1))
    
        
for yi, di in zip(te_pred, te_target):
	if(int(yi) == int(di)):
		c += 1                                                                    #Count correct predictions
	ct += 1                                                                     #Counter for total number of predictions

#Display the final accuracies in both training and testing datasets
print("\nNaive Bayes Accuracies")
print("---------------------")
print("Training Accuracy : {:.2f}%".format(c1*100/ct1))
print("Testing Accuracy : {:.2f}%".format(c*100/ct))