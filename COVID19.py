#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import quandl
import datetime


# In[2]:


df = pd.read_csv("Global_Mobility_Report.csv")


# In[3]:


len(df)


# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


df1 = df[['country_region_code', 'country_region',  'retail_and_recreation_percent_change_from_baseline',
       'grocery_and_pharmacy_percent_change_from_baseline',
       'parks_percent_change_from_baseline',
       'transit_stations_percent_change_from_baseline',
       'workplaces_percent_change_from_baseline',
       'residential_percent_change_from_baseline']]


# In[7]:


df2 = df1.groupby('country_region').mean()
df2


#     #Country Aggregate Data Starts From Here

# In[8]:


df2.to_csv("country_agg1.csv")


# In[9]:


df2 =pd.read_csv("country_agg1.csv", na_values=['#DIV/0!'])
df2


# In[10]:


len(df2)


# In[11]:


df2.reset_index(level=0, inplace=True)
df2['country_region'] = df2['country_region'].map(lambda name:name.strip().lower())


# In[12]:


df2


# In[13]:


data = pd.read_csv('Custom Data.csv', na_values=['#DIV/0!'])
data


# In[14]:


data =data.drop_duplicates()
data


# 

# In[15]:


data['Row Labels'] = data['Row Labels'] .map(lambda name: name.strip().lower())


# In[16]:


data = data.rename(columns = {'Row Labels':'country_region'})


# In[17]:



df2.dtypes


# In[18]:


out = pd.merge(df2,data,how='inner',on = 'country_region')
out


# In[19]:


out = out.fillna(0)
out = out.drop(columns=['index'])
out


# In[21]:


out.to_csv("final_Aggregate.csv");


# In[3]:


out = pd.read_csv("final_Aggregate.csv")
out


# In[4]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np


# In[8]:


# le = LabelEncoder()
# le.fit(out['country_region'].astype(str))
# out['country_region'] = le.transform(out['country_region'].astype(str))
out = out.drop(columns = ['Unnamed: 0'])
out = out.set_index('country_region')
# stdscalar = StandardScaler().fit_transform(out)

#mid = out.drop(columns = ['Unnamed: 0','country_region'])


# In[6]:


out.shape


# In[ ]:





# In[ ]:





# In[11]:


kmeans = KMeans(n_clusters=2)
kmeans.fit(out)


# In[12]:


kmeans.labels_


# In[13]:


out['cluster']=kmeans.labels_


# In[14]:


out.reset_index(level=0, inplace=True)


# In[15]:


import seaborn as sns


# In[ ]:


for


# In[16]:


out['cluster'].value_counts().index


# In[19]:


facet = sns.lmplot(data=out, x='Avg Income Gini coefficient', y='Average of Rule of Law', hue='cluster', 
                   fit_reg=False, legend=True, legend_out=True)


# In[110]:



facet2 = sns.catplot(x="country_region", y="cluster", hue="cluster", kind="bar", data=out,height=5, aspect=3)


# In[111]:


sns.pairplot(out)


# In[112]:


centers = kmeans.cluster_centers_


# In[113]:


plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


# In[78]:


from sklearn.cluster import DBSCAN


# In[100]:


clustering = DBSCAN(eps=15, min_samples=2).fit(out.drop(columns =['country_region','cluster']))


# In[101]:


clustering.labels_


# In[102]:


out['db_scan_cluster'] = clustering.labels_


# In[ ]:




