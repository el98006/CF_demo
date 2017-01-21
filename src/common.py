'''
Created on Jan 20, 2017

@author: eli
'''
import os
import csv
import numpy
import pandas

global rating

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
    data_dir = os.path.join(os.path.split(src_dir)[0],'ml-100k')
    rating_file = os.path.join(data_dir,'u.data')
    
    csv.register_dialect('tab', delimiter='\t')
    '''
    with open(rating_file,'rb') as fh:
        for row  in csv.reader(fh, dialect='tab'):         
            yield row
    '''
    names = ['uid', 'cid', 'rating', 'timestamp']
    raw_data = pandas.read_csv(rating_file, sep='\t', names=names)    
    raw_data    
    return raw_data


    
if __name__ == '__main__':
    raw_data = load_raw()
    n_users = raw_data['uid'].unique().shape[0]
    n_items = raw_data['cid'].unique().shape[0]
    
    ratings = numpy.zeros((n_users, n_items))
    for row in raw_data.itertuples():
        ratings[row[1]-1, row[2]-1] = row[3]
    
    print ratings
    

    
            
            