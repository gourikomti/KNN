#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


print(os.listdir())


# In[3]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


data = pd.read_csv('Social_Network_Ads.csv')


# In[5]:


data.head(10)


# In[6]:


data.info()


# In[7]:


# split your dataset
X = data.iloc[:, [2,3]].values
y = data.iloc[:, 4].values


# In[8]:


y


# In[9]:


#prepare data for test and training
from sklearn.model_selection import train_test_split


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[11]:


y_test


# In[12]:


X_test


# In[13]:


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[14]:


X_train


# In[15]:


#Move to KNN algorithm
from sklearn.neighbors import KNeighborsClassifier


# In[16]:


classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
classifier.fit(X_train, y_train)


# In[17]:


#get your predictor
y_predictor = classifier.predict(X_test)


# In[18]:


y_predictor


# In[19]:


#In classifiers, you will be creating a confusion matrix a lot
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predictor)


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


#for plotting a graph 


# In[22]:


from matplotlib.colors import ListedColormap
X_point, y_point = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_point[:, 0].min() - 1, stop = X_point[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_point[:, 1].min() - 1, stop = X_point[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_point)):
    plt.scatter(X_point[y_point == j, 0], X_point[y_point == j, 1],
                c = ListedColormap(('green', 'blue'))(i), label = j)
plt.title('K-NN Training set')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()


# In[23]:


X_point, y_point = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_point[:, 0].min() - 1, stop = X_point[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_point[:, 1].min() - 1, stop = X_point[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_point)):
    plt.scatter(X_point[y_point == j, 0], X_point[y_point == j, 1],
                c = ListedColormap(('green', 'blue'))(i), label = j)
plt.title('K-NN Training set')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()


# In[ ]:




