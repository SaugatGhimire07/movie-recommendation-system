#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np
import pandas as pd


# In[72]:


movies = pd.read_csv('dataset/movies.csv')
movie_ratings = pd.read_csv('dataset/ratings.csv')
tags = pd.read_csv('dataset/tags.csv')


# #### Movie Recommendation using Content Based Filtering

# In[73]:


movies.head()


# In[74]:


movie_ratings.head()


# In[75]:


tags.head()


# In[76]:


movies['genres'] = movies['genres'].str.replace('|',' ')
movies.head()


# In[77]:


len(movies.movieId.unique())


# In[78]:


#filtering out the users who have rated more than 25 movies
ratings_f = movie_ratings.groupby('userId').filter(lambda x: len(x) >= 25)

#list of the movies that remained pulled through the filtering
movies_list = ratings_f.movieId.unique().tolist()


# In[79]:


#filter the movies data frame
movies = movies[movies.movieId.isin(movies_list)]


# In[80]:


movies.head()


# In[81]:


#link movie to its id
Mapping_file = dict(zip(movies.title.tolist(), movies.movieId.tolist()))


# In[82]:


tags.drop(['timestamp'],1, inplace = True)
ratings_f.drop(['timestamp'],1, inplace = True)


# In[83]:


#create a merged dataframe of movies, genres and all the tags given to the movies
merged_dataset = pd.merge(movies, tags, on = 'movieId', how = 'left')
merged_dataset.head()


# In[84]:


#create metadata from tags and genres
merged_dataset.fillna("", inplace = True)
merged_dataset = pd.DataFrame(merged_dataset.groupby('movieId')['tag'].apply(
                                                        lambda x:"%s" % ' '.join(x)))
Final = pd.merge(movies, merged_dataset, on = 'movieId', how = 'left')
Final ['metadata'] = Final[['tag', 'genres']].apply(
                                            lambda x:' '.join(x), axis = 1)
Final[['movieId', 'title', 'metadata']].head()


# In[85]:


#Create count matrix from this new combined column
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
count_matrix = cv.fit_transform(Final["metadata"])


# In[86]:


# Now Compute the Cosine Similarity based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(count_matrix)


# In[87]:


# This Function takes movie title as input and return 15 most similar movies.
def get_recommendation(title):
    
    # Get the index of the movie that matches the title
    idx = Final['title'][Final['title']==title].index[0]
    
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x:x[1],reverse=True)
    
    # Get the scores of the 15 most similar movies
    sim_scores = sim_scores[0:16]
    
    for i in sim_scores:
        movie_index = i[0]
        print(Final['title'].iloc[movie_index])


# In[88]:


# Now lets make predictions
print('Movies Recommendation using Content Based Filtering')
get_recommendation("Toy Story (1995)")


# #### Movie Recommendation using Collaborative Filtering

# In[89]:


ratings = pd.read_csv('dataset/ratings.csv')
ratings = pd.merge(movies,ratings).drop(['genres','timestamp'],axis=1)
ratings.head()


# In[90]:


#create a merged dataframe of movies and all the ratings given to the movies by the users
userRatings = ratings.pivot_table(index=['userId'],columns=['title'],values='rating')
userRatings = userRatings.dropna(thresh=10, axis=1).fillna(0,axis=1)
#userRatings.fillna(0, inplace=True)
userRatings.head()


# In[91]:


#using cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(userRatings.T)

cosine_sim_df = pd.DataFrame(cosine_sim, index = userRatings.columns, columns =userRatings.columns)
cosine_sim_df.head(100)


# In[92]:


def get_similar(movie_name,rating):
    similar_ratings = cosine_sim_df[movie_name]*(rating-2.5)
    similar_ratings = similar_ratings.sort_values(ascending=False)
    #print(type(similar_ratings))
    return similar_ratings


# In[96]:

print('Movies Recommendation using Collaborative Filtering')
animated_movie_lover = [("Toy Story (1995)",5),("Moana (2016)",4),("Toy Story 3 (2010)",5),("2 Fast 2 Furious (Fast and the Furious 2, The) (2003)",1)]
similar_movies = pd.DataFrame()
for movie,rating in animated_movie_lover:
    similar_movies = similar_movies.append(get_similar(movie,rating),ignore_index = True)

similar_movies.head()
print(similar_movies.sum().sort_values(ascending=False).head(15))


# In[ ]:




