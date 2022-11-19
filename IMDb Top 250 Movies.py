#!/usr/bin/env python
# coding: utf-8

#                                            ABOUT THE DATASET
# 
# The dataset we’ve consisted of records of Top 250 movies rating by IMDB. I was able to webscrap the necessary informations and import them into an excel for prediction using the Clustering algorithm

# In[4]:


##IMPORT NECESSARY LIBARIES 
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')
from bs4 import BeautifulSoup
import requests
from csv import writer


# In[154]:


## GETTING THE URL OF THE WEBSITE 

source = requests.get("https://www.imdb.com/chart/top/")


# In[14]:


### WE USE RAISE_FOR_STATUS IN CASE THE WEBISTE IS NOT REACHABLE AT A POINT 

source.raise_for_status()


# In[155]:


### PASER IS USE TO ACCESS THE HTML PAGE OF THE SITE

soup = BeautifulSoup(source.text,'html.parser')


# In[16]:


soup()


# In[18]:


###GETTING THE TAG OF THE HTML
#### FIND_ALL() IS USED TO GET ALL THE NAMES IN THE TAG
movies = soup.find('tbody', class_="lister-list").find_all('tr')
print(len(movies))


# In[19]:


#### GETTING THE MOVIES NAME OF ROM THE HTML

for movie in movies:
    print(movie)
    break


# In[99]:


for movie in movies:
    Movie_name = movie.find('td', class_="titleColumn").a.text    ###EXTRACTING THE NAME OF THE MOVIE FROM THE HTML
    Movie_rank = movie.find('td', class_="titleColumn").get_text(strip=True).split('.')[0]  ###EXTRACTING THE RANK
    Movie_rating = movie.find('td', class_="ratingColumn").get_text(strip=True)  ##EXTRACTING THE RATING
    Movie_year = movie.find('td', class_="titleColumn").span.text.strip('()')   ###EXTRACTING THE YEAR 

    print(Movie_rank, Movie_name, Movie_year, Movie_rating)
    


# LOADING THE FILE INTO EXCEL

# In[115]:


import openpyxl


# In[131]:


excel = openpyxl.Workbook()
sheet = excel.active
sheet.title = 'IMDb Top 250 Movies'
print(excel.sheetnames)
sheet.append(["Movie Rank", "Movie Title", "Movie Year", " IMDBRating"]) ###CREATING HEADINGS FOR THE COLUMN NAMES

for movie in movies:
    Movie_name = movie.find('td', class_="titleColumn").a.text    ###EXTRACTING THE NAME OF THE MOVIE FROM THE HTML
    Movie_rank = movie.find('td', class_="titleColumn").get_text(strip=True).split('.')[0]  ###EXTRACTING THE RANK
    Movie_rating = movie.find('td', class_="ratingColumn").get_text(strip=True)  ##EXTRACTING THE RATING
    Movie_year = movie.find('td', class_="titleColumn").span.text.strip('()')   ###EXTRACTING THE YEAR 
    
    
    
    sheet.append([Movie_rank, Movie_name, Movie_year, Movie_rating])
excel.save("IMDb Top 250 Movies.xlsx")  ####SAVING THE EXCEL FILE TO THE LOCAL DIRECTORY


# In[161]:


###IMPORTING THE DATASET FROM DIRECTORY
dataset = pd.read_excel('C:/data/IMDb Top 250 Movies.xlsx')


# In[167]:


print(dataset.shape)
dataset.head()


# In[168]:


dataset.isna().sum()


# In[169]:


dataset.dtypes


# In[232]:


features = ['Movie Rank','Movie Year', ' IMDBRating']


# In[203]:


#Checking the distributions of the interactions

for feature in features:
    sns.distplot(dataset[feature]) 
    plt.show()


# In[186]:


print(dataset['Movie Year'].value_counts())


# In[204]:


print(dataset[' IMDBRating'].value_counts())


# In[184]:


##Correlation analysis in research is a statistical method used to measure the strength of the linear relationship between two variables and compute their association.


correlation = dataset.corr().transpose()
correlation


# In[185]:


plt.figure(figsize=(10,10))
sns.heatmap(correlation,vmax=1,vmin=-1,square=True)
plt.show()


# In[201]:


###SEGREGATE AND ZIPPING DATASET
IMDBRating = dataset[' IMDBRating'].values
Movie_Year = dataset['Movie Year'].values
x = np.array(list(zip(IMDBRating, Movie_Year)))
x


# In[205]:


x.shape


# In[206]:


###Let us now fit k-means algorithm on our scaled data and find out the optimum number of clusters to use

from sklearn.cluster import KMeans


# In[209]:


###FINDING THE OPTIMIZED K VALUE

sse = {} 
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000, random_state=1).fit(x)
    sse[k] = kmeans.inertia_

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()), color='r', marker='*')
plt.title('Optimal K Value')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# In[218]:


####MAKING PREDICTION USING KMEANS 
model=KMeans(n_clusters=2, random_state=42)
y_means = model.fit_predict(x)


# In[219]:


plt.scatter(x[y_means==0,0],x[y_means==0,1],s=50, c='yellow',label='1')
plt.scatter(x[y_means==1,0],x[y_means==1,1],s=50, c='blue',label='2')

plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1],s=100,marker='s', c='red')
plt.title('MOVIE RATING Analysis')
plt.xlabel(' IMDBRating')
plt.ylabel('Movie Year')
plt.show()


# In[220]:


###Converting the cluster to data frame 
convert = pd.DataFrame(y_means,columns=['convert']) 
convert


# # HIERARCHICAL CLUSTERING
# 

# In[237]:


dataset.head()


# In[264]:


datamovie = dataset.iloc[1:,2:]


# In[262]:


datamovie


# In[267]:


import scipy.cluster.hierarchy as clus

plt.figure(1, figsize=(16,8))
dendrogram = clus.dendrogram(clus.linkage(datamovie, method = "ward"))

plt.title("Dendrograme Tree Graph")
plt.xlabel('IMDB TOP 250 MOVIE RATING')
plt.ylabel('Distances')
plt.show()


# In[269]:


####FITTING THE HIERARCHIAL CLUSTERING TO THE DATASET WITH N=2

from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'average')
y_means = model.fit_predict(datamovie)
y_means


# ANALYSIS BASED ON THE FINDINGS
# 
# 1. WITH THE HELP OF CLUSTERING ALGORITHM WE WERE ABLE TO IDENTIFY THAT LEAST MOVIES WERE RELEASED BETWEEN THE YEAR OF 1920 -        1980 COMPARED TO 1980-2020
# 
# 2. MOVIES BETWEEN 1980 - 2020 HAD HIGHER RATING COMPARED TO 1920 - 1980
