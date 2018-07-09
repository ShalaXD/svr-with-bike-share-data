
# coding: utf-8

# In[35]:


import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import random


# #############################################################################
# Generate sample data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

print(X.shape, y.shape)
# #############################################################################
# Add noise to targets
# y[::5] += 3 * (0.5 - np.random.rand(8))

# #############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

# #############################################################################
# Look at the results
lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()


# In[ ]:





# In[49]:


from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.svm import SVR

my_data = genfromtxt('bitcoin.csv', delimiter=',')
y = np.reshape(my_data[1:,1:],1772)
X = np.arange(1772)
X = np.reshape(X,(1772,1))

print(data)
# plt.plot(X,y)
# plt.show()
print(X.shape, y.shape)


svr_rbf = SVR(kernel='rbf', C=1000, gamma=0.1)
y_rbf = svr_rbf.fit(X, y).predict(X)
lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')

plt.show()


# In[80]:


data = genfromtxt('day.csv', delimiter=',')
X = data[1:,[2,3,4,5,6,7,8,9,10,11,12]]
print(X.shape)
y = data[1:,15]
print(y.shape)



# In[81]:


import random
import math


idx = random.sample(range(X.shape[0]), math.floor(X.shape[0]*0.2))
test_x = X[idx,:]
print("test_x: ", test_x.shape)
train_x = np.delete(X, idx, axis = 0)
print("train_x: ", train_x.shape)


test_y = y[idx]
print("test_y: ", test_y.shape)
train_y = np.delete(y, idx)
print("train_y: ", train_y.shape)


# In[108]:


c_list = [1,10,100,1000, 5000, 10000]
eps_list = [0.01, 0.1, 1, 10, 100, 1000]

for c in c_list:
    for eps in eps_list:
        print(c, eps)
        svr_rbf = SVR(kernel='rbf', C=c, epsilon = eps, gamma = 0.1)
        model = svr_rbf.fit(train_x, train_y)
        score_rbf = model.score(test_x,test_y)
        print(score_rbf)
        y_rbf = model.predict(test_x)
        plt.plot(test_y,label='Actual value')
        plt.plot(y_rbf,label='Predicted value')
        plt.xlabel('Test data sample')
        plt.ylabel('Predicted and actural values')
        plt.title('Support Vector Regression result')
        plt.legend()
        plt.show()


# In[ ]:


c_list = [1000, 5000, 10000]
eps_list = [0.01, 0.1, 1, 10, 100, 1000]

for c in c_list:
    for eps in eps_list:


# In[104]:


from sklearn.decomposition import PCA
pca_data = PCA(n_components=1).fit_transform(train_x)
# print(pca_data)
plt.scatter(pca_data, train_y)
plt.show()


# In[ ]:




