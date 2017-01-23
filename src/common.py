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

class user:
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
        
        
def load_raw():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    print os.path.dirname(__file__)
    data_dir = os.path.join(os.path.split(src_dir)[0],'ml-100k')
    rating_file = os.path.join(data_dir,'u.data')
    item_file = os.path.join(data_dir,'u.item')
    raw_item = pandas.read_csv(item_file, sep='|',header=None)
    
    header = ['uid', 'cid', 'rating', 'timestamp']
    raw_data = pandas.read_csv(rating_file, sep='\t', names=header)    
    return raw_data, raw_item

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, numpy.newaxis]) 
        pred = mean_user_rating[:, numpy.newaxis] + similarity.dot(ratings_diff) / numpy.array([numpy.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / numpy.array([numpy.abs(similarity).sum(axis=1)])     
    return pred    

def  predict_for_user(rating_matrix, uid):
    predictions = rating_matrix.loc[uid-1,R.loc[uid-1,:] == 0]     
    
    
if __name__ == '__main__':
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
    
    for i in range(10):
        cid = random.randrange(0,n_users)
        top_5 = item_prediction[cid].argsort()[-5:][::-1]
        for i in top_5:
            print raw_item[1][int(i)-1]
        print '-=-=-=-=-=-='