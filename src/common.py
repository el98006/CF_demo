'''
Created on Jan 20, 2017

@author: eli
'''
import os
import numpy
import pandas
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import cross_validation as cv

import random 

class User:
#user id | age | gender | occupation | zip code
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def __repr__(self):
        return 'id:{}, age:{}, gender:{} occupation:{} \n'.format(self.id, self.age,self.gender,self.occupation) 
    
    def __eq__(self, u):
        if self.id == u.id:
            return True
        else: 
            return False
class Movie:
    __slot__ =   ['id', 'title', 'release_date','video release date','url','genre']       
    def __init__(self, *args):
        self.__slot__ = args            
        
    def __repr(self):
        return 'name:{}'.format(self.title)
    
    def __cmp__ (self, other):
        return cmp(self.id,other.id)


class ItemDB(object):
    
    movie_collection = {}
    
    def __init__(self, full_path_name):
        headers = ['id', 'title', 'release_date','video release date',\
        'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation','Children\'s','Comedy',\
        'Crime','Documentary', 'Drama','Fantasy','Film-Noir','Horror','Musical', 'Mystery', 'Romance', 'Sci-Fi'\
        'Thriller','War','Western']
        
        raw_item = pandas.read_csv(full_path_name, sep='|',header=headers )
        
        for item in raw_item:
            genre_idx = item[7:].nonzero()
            genre = numpy.array(headers[7:])[genre_idx]
            self.movie_collection[item[0]] = Movie(item[1:6],genre) 
        
    def get_item_by_id(self, cid):
        try:
            return self.movie_collection[cid]
        except KeyError:
            print "cid doesn't exist"
    
        
def load_raw():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.split(src_dir)[0],'ml-100k')
    rating_file = os.path.join(data_dir,'u.data')
    item_file = os.path.join(data_dir,'u.item')
    raw_item = pandas.read_csv(item_file, )
    
    header = ['uid', 'cid', 'rating', 'timestamp']
    raw_data = pandas.read_csv(rating_file, sep='\t', names=header)    
    return raw_data, raw_item

def predict(ratings, similarity, type='user'):
    if type == 'user':
        # axis index of rows/y is 0, 1 for columns/x.  
        # mean_user_rating is the mean of all values in the same row.
        mean_user_rating = ratings.mean(axis=1)
     
        # rating_diff is the 3rd dimension, axis=2
        ratings_diff = (ratings - mean_user_rating[:, numpy.newaxis]) 
        
        pred = mean_user_rating[:, numpy.newaxis] + similarity.dot(ratings_diff) / numpy.array([numpy.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / numpy.array([numpy.abs(similarity).sum(axis=1)])     
    return pred    


    
def  predict_for_user(rating_matrix, uid):
    # predict against items which has no ratings 
    predictions = rating_matrix.loc[uid-1,:] == 0     

'''    
def top_n_nearest(item_similarity, cid, top_n=5):
    
    sorted_idx = item_similarity[cid].argsort()[0-top_n:][::-1]  
        # [::-1] reverse order for all the values
        
    reco_list = map(get_item_details, sorted_idx)
    return reco_list
'''
        
if __name__ == '__main__':
    
    m_db =  ItemDB()
    my_m = m_db.get_item_by_id('23')
    
    '''
    raw_data, raw_item = load_raw()

    n_users = raw_data['uid'].unique().shape[0]
    n_items = raw_data['cid'].unique().shape[0]

    train_data, test_data = cv.train_test_split(raw_data, test_size=0.25)
            
    train_data_matrix = numpy.zeros((n_users, n_items))
    for line in train_data.itertuples():
        train_data_matrix[line[1]-1, line[2]-1] = line[3]  

    test_data_matrix = numpy.zeros((n_users, n_items))
    for line in test_data.itertuples():
        test_data_matrix[line[1]-1, line[2]-1] = line[3]       
        
    #user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
    item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')        
    
    item_prediction = predict(train_data_matrix, item_similarity, type='item')
    
    for i in range(5):
        cid = random.randrange(0,n_items)
        top_n_nearest(item_similarity, cid)
    '''