#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[2]:


gold_data = pd.read_csv('dataset.csv')


# In[3]:


gold_data.head()


# In[4]:


gold_data.tail()


# In[5]:


gold_data.shape


# In[6]:


gold_data.info()


# In[7]:


gold_data.isnull().sum()


# In[8]:


gold_data.describe()


# In[10]:


correlation = gold_data.corr(numeric_only=True)


# In[16]:


plt.figure(figsize = (7,7))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':7}, cmap='Reds')


# In[17]:


# correlation values of GLD
print(correlation['GLD'])


# In[22]:


# checking the distribution of the GLD Price
sns.distplot(gold_data['GLD'],color='blue')


# In[23]:


X = gold_data.drop(['Date','GLD'],axis=1)
Y = gold_data['GLD']


# In[24]:


print(X)


# In[25]:


print(Y)


# In[27]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)


# In[28]:


regressor = RandomForestRegressor(n_estimators=100)


# In[29]:


regressor.fit(X_train,Y_train)


# In[30]:


test_data_prediction = regressor.predict(X_test)


# In[31]:


print(test_data_prediction)


# In[32]:


error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)


# In[33]:


Y_test = list(Y_test)


# In[35]:


plt.plot(Y_test, color='red', label = 'Actual Value')
plt.plot(test_data_prediction, color='blue', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()


# In[ ]:




