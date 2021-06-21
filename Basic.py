
# coding: utf-8

# In[70]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import sklearn


# In[123]:


#df = pd.read_csv("heart.csv")


# In[130]:


arr = np.array([[1, 0, 0, 0, 0],[1, 0, 0, 0, 0],[1, 0, 0, 0, 0],[1, 0, 0, 0, 0],[1, 0, 0, 0, 0],[1, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],[1, 0, 0, 0, 0],[1, 0, 0, 0, 0],[1, 0, 0, 0, 0],[1, 0, 0, 0, 0],[1, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[1, 0, 0, 0, 0],
[1, 0, 0, 0, 0],[1, 0, 0, 0, 0],[1, 0, 0, 0, 0],[1, 0, 0, 0, 0],[1, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[1, 0, 0, 0, 0],[1, 0, 0, 0, 0],
[1, 0, 0, 0, 0],[1, 0, 0, 0, 0],[1, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]])

column=["10th","12th","graduation","entrance","Final_result"]

df = pd.DataFrame(arr,columns = column)


# In[131]:


df.head()


# In[45]:


#df.values
#df.columns
#df.describe


# In[46]:


#df.output.sum()
#df.output.count()


# In[57]:


#df.isnull().sum()


# In[58]:


#x_corr = df.corr()


# In[62]:


x_var = df.var()


# In[66]:


#x_var


# In[67]:


#df.restecg.var()


# In[68]:


x=df.iloc[:,:-1]     #input data


# In[69]:


y=df.iloc[:,-1]     #outputdata


# In[83]:


#from sklearn.imputs import SimpleImputer
#imputs = SimpleImputer(missing_values=np.nan , strategy = "mean")
#imputs.fit_transform(df)


# In[84]:


#from sklearn.preprocessing import LableEncoder
#lable = LableEncoder()
#lable.fit_transform(df)


# In[85]:


#from sklearn.compose import columnTransformer
#col = columnTransformer(['encoder',OneHotEncoder(),[0]],remainder="passthrough)
#x = np.array(col.fit_transform(df),dtype="str")

#or

#pd.get_dummies(df["col names"])


# #Train_Test_Split

# In[88]:


#from sklearn.model_selection import train_test_split
# x_train , y_train , x_test , y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# #Standardisation

# In[91]:


#from sklearn.preprocessing import StandardScalar 
#sc= StandardScalar()
#sc.fit(x_train)
#sc.fit(y_train)

