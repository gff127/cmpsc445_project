#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import pandas as pd


# In[53]:


testData = pd.read_csv(r"C:\Users\mitch\Desktop\CMPSC445\GP\test.csv")
trainData = pd.read_csv(r"C:\Users\mitch\Desktop\CMPSC445\GP\train.csv")


# In[54]:


testData.head()


# In[55]:


trainData.head()


# In[56]:


testData = np.asarray(testData)
trainData = np.asarray(trainData)


# In[57]:


testData


# In[58]:


testData = np.delete(testData, 0, 1)
testData = np.delete(testData, 0, 1)
testData = np.delete(testData, 5, 1)
testData = np.delete(testData, 6, 1)
testData = np.delete(testData, 6, 1)
testData = np.delete(testData, 13, 1)


# In[59]:


testData


# In[60]:


trainData = np.delete(trainData, 0, 1)
trainData = np.delete(trainData, 0, 1)
trainData = np.delete(trainData, 5, 1)
trainData = np.delete(trainData, 6, 1)
trainData = np.delete(trainData, 6, 1)
trainData = np.delete(trainData, 13, 1)


# In[61]:


trainData


# In[62]:


testDataY = testData[:,-1]


# In[63]:


testDataX = testData[:,:-1]


# In[64]:


trainDataY = trainData[:,-1]


# In[65]:


trainDataX = trainData[:,:-1]


# In[66]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 50)
knn.fit(trainDataX, trainDataY)


# In[67]:


knn.score(testDataX, testDataY)


# In[68]:


from sklearn.metrics import confusion_matrix
dataYPred = knn.predict(testDataX)
confusionMatrix = confusion_matrix(testDataY, dataYPred)
confusionMatrix


# In[69]:


sensitivity = 21338/(21338+807)
specificity = 2825/(2825+3468)
precision = 21338/(21338+3468)
accuracy = (21338+2825)/(21338+2825+807+3468)
print(sensitivity)
print(specificity)
print(precision)
print(accuracy)


# In[86]:


import matplotlib.pyplot as plt
marker_size=15
plt.scatter(testDataX[:,0], testDataX[:,1], marker_size, c=testDataX[:,2])
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




