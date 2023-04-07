#import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#load the csv using pandas
csvData = pd.read_csv('/home/linuxdev/coding/heart.csv')
csvData.head()

csvData.dropna(inplace = True)

#define features and target 
X = csvData.drop(['target'], axis =1 )
y = csvData['target']
# train the model using logistic regression
#algorithm
#from the scikit learn library
model_logreg = LogisticRegression()
model_logreg.fit(X, y)

y_pred = model_logreg.predict(X)

#verify the accuracy of the model


import pickle
pickle.dump(model_logreg,open('model_logreg.pkl','wb'))





