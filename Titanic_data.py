#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## The Data
# 
# Reading in the titanic_train.csv file into a pandas dataframe.

# In[22]:


train = pd.read_csv('titanic_train.csv')


# In[23]:


train.head()


# # Exploratory Data Analysis
# 
# ## Missing Data
# 
# Using seaborn to create a simple heatmap to see where the data are missing!

# In[24]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. I drop this later.

# In[25]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')


# In[26]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[27]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[28]:


sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)


# In[29]:


train['Age'].hist(bins=30,color='darkred',alpha=0.7)


# In[30]:


sns.countplot(x='SibSp',data=train)


# In[31]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# ___
# ## Data Cleaning
# Filling in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation).
# 
# However I am filling the average age by passenger class.
# 

# In[32]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# The wealthier passengers in the higher classes tend to be older, which makes sense. I'll use these average age values to impute based on Pclass for Age.

# In[33]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# Now apply that function!

# In[34]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# Now let's check that heat map again!

# In[35]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Now dropping the Cabin column.

# In[36]:


train.drop('Cabin',axis=1,inplace=True)


# In[37]:


train.head()


# In[38]:


train.dropna(inplace=True)


# ## Converting Categorical Features 
# 
# Converting categorical features to dummy variables using pandas! Otherwise machine learning algorithm won't be able to directly take in those features as inputs.

# In[39]:


train.info()


# In[40]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[41]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[42]:


train = pd.concat([train,sex,embark],axis=1)


# In[43]:


train.head()


# Data is ready for model!
# 
# # Building a Logistic Regression model
# 
# Splitting data into a training set and test set
# 
# ## Train Test Split

# In[44]:


from sklearn.model_selection import train_test_split


# In[63]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), train['Survived'], test_size=0.45)


# ## Training and Predicting

# In[64]:


from sklearn.linear_model import LogisticRegression


# In[65]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[66]:


predictions = logmodel.predict(X_test)


# ## Evaluation

# Checking precision,recall,f1-score using classification report!

# In[67]:


from sklearn.metrics import classification_report


# In[68]:


print(classification_report(y_test,predictions))


# ## Thank You!
