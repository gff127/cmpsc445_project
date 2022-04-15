#preprocessing the data
#includes fixing missing values, normalization, feature selection,
#and splitting  testing/training data into separate files
import numpy
import numpy as np
import pandas
import math
import datetime

#file name of the raw data
FILE_NAME = "rawData.csv"




categorical_features = []
numeric_features = []
data = pandas.read_csv(FILE_NAME) #data from the .csv file
columns = data.columns #column names
target = columns[-1] #target variable


    
#delete rows where the target variable is missing 
data = data[data[target] == data[target]]



#sort features into categorical and numeric
for i in columns:
        if type(data[i][1]) == type("") and i != target:
            categorical_features.append(i)
        elif i != target:
            numeric_features.append(i)


    




#for tracking how many missing values each feature has
num_missing = {}

#replace missing values in numeric features
#and normalize the data
for i in numeric_features:
    #NaN is replace with the mean value
    mean = numpy.mean(data[i])
    num_missing[i] = data[i].isna().sum()
    data[i] = data[i].fillna(mean)
    
    #min max normalization
    min_val = data[i].min()
    max_val = data[i].max()
    data[i] = (data[i] - min_val)/(max_val - min_val)


#replace missing values in categorical features
for i in categorical_features:
    #the most common value is used
    mode = (data[i].mode(dropna = "True"))[0]
    data[i] = data[i].fillna(mode)




corr_matrix = data.corr(method = "pearson") #pearson correlation matrix
dim = len(numeric_features) #dimension of the matrix

#the point at which 2 features are considered strongly correlated
STRONG_CORRELATION = 0.75

redundant_features = set()
#find redundant numerical features
for i in range(dim):
    for j in range(dim):
        #if the features are redundant
        if i > j and abs(corr_matrix[numeric_features[i]][numeric_features[j]]) > STRONG_CORRELATION:
            #if one of the features hasn't already been chosen to be removed
            if i not in redundant_features and j not in redundant_features:
                #prioritize features with more missing data to be removed
                if num_missing[numeric_features[i]] > num_missing[numeric_features[j]]:
                    redundant_features.add(numeric_features[i])
                else:
                    redundant_features.add(numeric_features[j])
#remove redundant features               
data = data.drop(columns = redundant_features)




#convert the full date into just the month
data["Date"] = pandas.to_datetime(data["Date"]).dt.month_name()
data = data.rename(columns = {"Date": "Month"})



#shuffle the order of rows
#the randomized order will the same each time the code is ran
data = data.sample(frac = 1, random_state = 1).reset_index()
data = data.drop(columns = {"index"})



TESTING_PERCENT = 0.20 #the percentage of data used for testing
#split data into testing and training data
cutoff = math.floor(TESTING_PERCENT * data.shape[0])
testing_data = data[0:cutoff]
training_data = data[cutoff:]

print(data["Evaporation"].mean())

#write the training/testing data to .csv files
testing_data.to_csv("test.csv", index = False)
training_data.to_csv("train.csv", index = False)
