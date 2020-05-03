from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sklearn
from geopy.geocoders import Nominatim
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING finalDATA.py 
import finalData as fD


# DATA COLLECTING
location = None
while not location:
  # userLoc = input()
  userLoc = 'Chicago, IL'
  geolocator = Nominatim(user_agent="test1")
  location = geolocator.geocode(userLoc)
  if not location:
    print('invalid location, pls try again')

# userAge = input()
userAge = 21

# get closest county to my coords
closestCtyOb = fD.nearest((location.longitude, location.latitude))

# print prob of getting the infected in the county
city1 = fD.prob(closestCtyOb)


def countyFrame(CNTY) :
    data = {
        'CNTY NAME' : CNTY['CTNY'],
        'AGE RANGE' : CNTY['AGE RANGE'],
        'TOT POP BY AGE GROUP' : CNTY['TPOP PA PCN'],
        'CASES BY AGE GROUP' : CNTY['Number of Cases Per Age Range'],
        'MOBILITY TOTAL' : CNTY['MOBILITY TOTAL']
    }

    df = pd.DataFrame(data)
    df['PRECENT CASES BY AGE GROUP'] = (df['CASES BY AGE GROUP'] / df['TOT POP BY AGE GROUP'])*100

    #OUTBREAK THRESHOLD - can be described as 1 / Log(R0) -- R0 is the Basic Repoduction number, i.e. the number of people infected by one case
    #R0 number as of April 7, 2020 reported as 2.2 - 2.7 (2.4)
    outbreak_threshold = 1 / (math.log(5.7))
    #print(outbreak_threshold)



    df['IsOutbreak'] = df['PRECENT CASES BY AGE GROUP'] >= outbreak_threshold

    return df


df1 = countyFrame(city1)

#BUIDLING TESTING DATA SET


path = '/Users/connormahern/Desktop/AI_Final_Project/'
sexDistData = path + 'AI_FinalProject/uscities.csv'

#Single City Test Cases
'''
city = 'Evansville, IN'

geolocator1 = Nominatim(user_agent="test1")
location1 = geolocator1.geocode(city)
closestCtyOb = fD.nearest((location1.longitude, location1.latitude))

cityProb = fD.prob(closestCtyOb)
cityProb = countyFrame(cityProb)

print(cityProb)


'''
dfCities = pd.read_csv(sexDistData, usecols = ['city','state_id'])


frames = []

for elm in dfCities.iterrows():

    try :
        city = elm[1][0] + ', ' + elm[1][1]
        print(city)
        
        geolocator1 = Nominatim(user_agent="test1")
        location1 = geolocator1.geocode(city)
        closestCtyOb = fD.nearest((location1.longitude, location1.latitude))
        #print(closestCtyOb)

        cityProb = fD.prob(closestCtyOb)
        cityProb = countyFrame(cityProb)

        frames.append(cityProb)
        index += 1

    except :
        continue

df2 = pd.concat(frames)
normalize_mob = df2['MOBILITY TOTAL']
print(type(normalize_mob))
df2 = df2.drop(['MOBILITY TOTAL'], axis=1 )



min_max_scaler = sklearn.preprocessing.MinMaxScaler()
normalize_mob = normalize_mob.to_numpy()
normalize_mob = normalize_mob.reshape(-1, 1)
mob_scaled = min_max_scaler.fit_transform(normalize_mob)

mob_scaled = pd.DataFrame(mob_scaled)

df2 = df2.join(mob_scaled)

ids = df2[['CNTY NAME', 'AGE RANGE']]
y = df2['IsOutbreak']
X = df2.drop(['CNTY NAME', 'AGE RANGE', 'IsOutbreak'], axis=1)
print(X)
print(y)



#modeling

model = LogisticRegression(random_state=42)
model.fit(X, y)

y_hat = pd.Series(model.predict(X), name='Pred IsOutbreak')


print(accuracy_score(y, y_hat))

#checking the model with R^2 - 

#print(len(ids))
#print(len(y))
#print(len(y_hat))


output_df = pd.concat([ids, y, y_hat], axis=0)
#print(output_df.head())



outbreak_prob = pd.DataFrame(model.predict_proba(X), columns=['Prob Not Outbreak', 'Pred Prob to be Outbreak']) 
#print(outbreak_prob.head())


#df2 = df2.drop(['PRECENT CASES BY AGE GROUP', 'IsOutbreak', 'TOT POP BY AGE GROUP', 'CASES BY AGE GROUP'],  axis = 1)
#outbreak_prob = outbreak_prob.drop(['Prob Not Outbreak'], axis = 1)

df2.reset_index(drop=True, inplace=True)
outbreak_prob.reset_index(drop=True, inplace=True)

print(df2.head)
print(outbreak_prob)

# result = pd.concat([df2, outbreak_prob], axis=1, sort=False)
result = df2.join(outbreak_prob)
result.to_csv('results1l.csv',index= True)
print(result)







