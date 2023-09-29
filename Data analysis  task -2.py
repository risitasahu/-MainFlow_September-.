#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


df = pd.read_csv("INDIAvi....csv")
df


# In[5]:


df.info


# In[7]:


df.isnull()


# In[10]:


df.fillna(method='ffill',inplace=True)


# In[11]:


df.isnull().sum()


# In[12]:


df.columns


# In[13]:


df.isnull().sum()


# In[14]:


df.info


# In[15]:


df.duplicated()


# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("INDIAvi....csv")

# Step 2: Define a function to remove outliers using the IQR method
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
df[numeric_columns].boxplot(vert=False)
plt.title("Boxplots of Numerical Columns (Before Outlier Removal)")
plt.xlabel("Value")
for column in numeric_columns:
    df = remove_outliers_iqr(df, column)
plt.subplot(1, 2, 2)
df[numeric_columns].boxplot(vert=False)
plt.title("Boxplots of Numerical Columns (After Outlier Removal)")
plt.xlabel("Value")
plt.tight_layout()
plt.show()


# In[20]:


import pandas as pd
df = pd.read_csv("INDIAvi....csv")
summary_stats = df.describe()
print(summary_stats)


# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("INDIAvi....csv")


correlation_matrix = df.corr()


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(df['views'], df['likes'], alpha=0.5)
plt.title("Scatter Plot: views vs likes")
plt.xlabel("Views")
plt.ylabel("Likes")


plt.subplot(1, 2, 2)
plt.scatter(df['dislikes'], df['comment_count'], alpha=0.5)
plt.title("Scatter Plot: dislikes vs comment_count")
plt.xlabel("Dislikes")
plt.ylabel("Comment Count")

plt.tight_layout()
plt.show()


# In[22]:


import pandas as pd
import scipy.stats as stats


correlation, p_value = stats.pearsonr(df['likes'], df['views'])

alpha = 0.05  # significance level
if p_value < alpha:
    print("The correlation is statistically significant.")
    if correlation > 0:
        print("There is a positive correlation between likes and views.")
    else:
    print("There is a negative correlation between likes and views.")
else:
    print("The correlation is not statistically significant.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




