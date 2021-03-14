#!/usr/bin/env python
# coding: utf-8

# # Graduate Rotational Internship Program The Sparks Foundation

# ## Data Science & Business Analytics Task - 1

# ## Problem statement

# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# To predict:
# What will be predicted score if a student studies for 9.25 hrs/ day?

# ### Author : Ramanathan N

# ### Importing the Libraries

# In[1]:


import numpy as np  
import pandas as pd
import lux #new data visualization library

import warnings 
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


path = "D:\Ram N\Documents\student_scores.csv"
data = pd.read_csv(path)


# ### Understanding the DataSet

# In[3]:


data.head()


# In[4]:


data.isnull()


# In[5]:


data.describe()


# In[6]:


data.columns


# In[7]:


data.shape


# In[8]:


train,test = train_test_split(data,test_size=0.25)


# In[9]:


train.shape


# In[10]:


test.shape


# ### Preparing Data for Modelling

# In[11]:


train_x=train.drop("scores",axis=1)
train_y=train["scores"]


# In[12]:


test_x=test.drop("scores",axis=1)
test_y=test["scores"]


# In[13]:


lr=LinearRegression()


# In[14]:


lr.fit(train_x,train_y)


# In[15]:


lr.coef_


# In[16]:


lr.intercept_


# In[17]:


# Plotting the regression line # formula for line is y=m*x + c
line = lr.coef_*train_x+lr.intercept_
# Plotting for the test data
plt.scatter(train_x,train_y)
plt.plot(train_x, line);
plt.show()


# ### Making Prediction

# In[18]:


pr=lr.predict(test_x)


# In[19]:


list(zip(test_y,pr))


# In[20]:


df=pd.DataFrame({'Actual values' : test_y, 'Predicted values' :pr })
df


# In[21]:


from sklearn.metrics import mean_squared_error


# In[22]:


mean_squared_error(test_y,pr,squared=False)


# ### Testting the model to redict the percentage of student if he studies for 9.25 hours as given

# In[23]:


hour =[9.25]
own_pr=lr.predict([hour])
print("No of Hours = {}".format([hour]))
print("Predicted Score = {}".format(own_pr[0]))


# #### Conclusion: Here root mean squared error is less than 10% of the mean value of the percenttages of all the student scores. Hence, it is safe to conclue the model did a decent job to predict the the student score as 89.60 % when student studies for 9.25 hours.

# In[ ]:




