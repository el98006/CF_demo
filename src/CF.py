'''
Created on Jan 20, 2017

@author: eli
'''

global rating 
def pearson_correlation(u1,u2):

    # To get both rated items
    both_rated = {}
    for item in rating[u1]:
        if item in rating[u2]:
            both_rated[item] = 1

    number_of_ratings = len(both_rated)        
    
    # Checking for number of ratings in common
    if number_of_ratings == 0:
        return 0

    # Add up all the preferences of each user
    sum_1 = sum([rating[u1][item] for item in both_rated])
    sum_2 = sum([rating[u2][item] for item in both_rated])

    rating[u1][item] - sum1 for item in both_rated
    # Sum up the squares of preferences of each user
    person1_square_preferences_sum = sum([pow(dataset[person1][item],2) for item in both_rated])
    person2_square_preferences_sum = sum([pow(dataset[person2][item],2) for item in both_rated])

    # Sum up the product value of both preferences for each item
    product_sum_of_both_users = sum([dataset[person1][item] * dataset[person2][item] for item in both_rated])

    # Calculate the pearson score
    numerator_value = product_sum_of_both_users - (person1_preferences_sum*person2_preferences_sum/number_of_ratings)
    denominator_value = sqrt((person1_square_preferences_sum - pow(person1_preferences_sum,2)/number_of_ratings) * (person2_square_preferences_sum -pow(person2_preferences_sum,2)/number_of_ratings))
    if denominator_value == 0:
        return 0
    else:
        r = numerator_value/denominator_value
        return r 

def most_similar_users(person,number_of_users):
    # returns the number_of_users (similar persons) for a given specific person.
    scores = [(pearson_correlation(person,other_person),other_person) for other_person in dataset if  other_person != person ]
    
    # Sort the similar persons so that highest scores person will appear at the first
    scores.sort()
    scores.reverse()
    return scores[0:number_of_users]

def user_reommendations(person):

    # Gets recommendations for a person by using a weighted average of every other user's rankings
    totals = {}
    simSums = {}
    rankings_list =[]
    for other in dataset:
        # don't compare me to myself
        if other == person:
            continue
        sim = pearson_correlation(person,other)
        #print ">>>>>>>",sim

        # ignore scores of zero or lower
        if sim <=0: 
            continue
        for item in dataset[other]:

            # only score movies i haven't seen yet
            if item not in dataset[person] or dataset[person][item] == 0:

            # Similrity * score
                totals.setdefault(item,0)
                totals[item] += dataset[other][item]* sim
                # sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+= sim

        # Create the normalized list

    rankings = [(total/simSums[item],item) for item,total in totals.items()]
    rankings.sort()
    rankings.reverse()
    # returns the recommended items
    recommendataions_list = [recommend_item for score,recommend_item in rankings]
    return recommendataions_list
if __name__ == '__main__':
    pass