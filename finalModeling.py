from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sklearn
from geopy.geocoders import Nominatim
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import pickle


#IMPORTING finalData which feeds data from pipeline
import getData as fD

# structure to create a df for an individual county
def countyFrame(CNTY) :
    data = {
        'CNTY NAME' : CNTY['CTNY'],
        'AGE RANGE' : CNTY['AGE RANGE'],
        'TOT POP BY AGE GROUP' : CNTY['TPOP PA PCN'],
        'CASES BY AGE GROUP' : CNTY['Number of Cases Per Age Range'],
        'MOBILITY TOTAL' : CNTY['MOBILITY TOTAL']
    }

    # create county frame
    df = pd.DataFrame(data)

    # calc percentage of age roup thats infected
    df['PRECENT CASES BY AGE GROUP'] = (df['CASES BY AGE GROUP'] / df['TOT POP BY AGE GROUP'])*100

    #OUTBREAK THRESHOLD - can be described as 1 / Log(R0) -- R0 is the Basic Repoduction number, i.e. the number of people infected by one case
    #R0 number as of April 7, 2020 reported as 5.7
    outbreak_threshold = 1 / (math.log(5.7))
    df['IsOutbreak'] = df['PRECENT CASES BY AGE GROUP'] >= outbreak_threshold

    return df


#BUILDING RESULTS

frames = []

# pull calcualted counties from pickle file for faster computing
with open('frames.pickle', 'rb') as f:
    frames = pickle.load(f)
df2 = pd.concat(frames)
df2.reset_index(drop=True, inplace=True)

#Normalizing Mobility Data
prenormalized_mob = df2['MOBILITY TOTAL']
normalize_mob = df2['MOBILITY TOTAL']
df2 = df2.drop(['MOBILITY TOTAL'], axis=1 )
#using the minmax normmalize from sklearn to normalize our mobility data on a scale of 1 - 0 (ex .5)
min_max_scaler = sklearn.preprocessing.MinMaxScaler()
normalize_mob = normalize_mob.to_numpy()
normalize_mob = normalize_mob.reshape(-1, 1)
mob_scaled = min_max_scaler.fit_transform(normalize_mob)
mob_scaled = pd.DataFrame(mob_scaled)
df2 = df2.join(mob_scaled)
df2.columns = ['CNTY NAME','AGE RANGE','TOT POP BY AGE GROUP', 'CASES BY AGE GROUP','PRECENT CASES BY AGE GROUP','IsOutbreak', 'Mobility Total']
df2 = df2.fillna(0)

# incorporate mobility data into model
df2['PRECENT CASES BY AGE GROUP'].loc[(df2['Mobility Total'] > .7)] = df2['PRECENT CASES BY AGE GROUP'] / .9
df2['IsOutbreak'].loc[(df2['PRECENT CASES BY AGE GROUP'] > (1 / (math.log(5.7))))] = True

#CLASSIFYING TESTING AND TRANING DATA
traing_data = df2.head(n=2000)
testing_data = df2.tail(n=1000)


#Creating model X and Y from traning
ids_training = traing_data[['CNTY NAME', 'AGE RANGE', 'PRECENT CASES BY AGE GROUP', 'Mobility Total']]
y = traing_data['IsOutbreak']
x = traing_data.drop(['CNTY NAME', 'AGE RANGE', 'IsOutbreak'], axis=1)

#modeling
model = LogisticRegression(random_state=42)
model.fit(x, y)


#GETTING TRAINING DATA AND COMPLINING INTO ONE DF
outbreak_prob = pd.DataFrame(model.predict_proba(x), columns=['Prob Not Outbreak', 'Predicted Prob Outbreak']) 
traing_result = ids_training.join(outbreak_prob)
#print(traing_result)

#This is our accary score
y_hat = pd.Series(model.predict(x), name='Pred IsOutbreak') 
print("Accuracy Score : " + str(accuracy_score(y, y_hat)))

#TESTING DATA OR Y_HAT
ids_testing = testing_data[['CNTY NAME', 'AGE RANGE', 'PRECENT CASES BY AGE GROUP', 'Mobility Total']]
testing_data = testing_data.drop(['CNTY NAME', 'AGE RANGE', 'IsOutbreak'], axis=1)
ids_testing.reset_index(drop=True, inplace=True)
testing_result = pd.DataFrame(model.predict_proba(testing_data), columns=['Prob Not Outbreak', 'Predicted Prob Outbreak'])
testing_result = ids_testing.join(testing_result)
#print(testing_result)





#USER INPUT SECTION
def userInput(age, cnty, state) :
    try :
        cntyObj = 0

        #Looking for county and state in our pickle frame
        for obj in frames :
            if cnty in obj.get('CNTY NAME')[0] and state in obj.get('CNTY NAME')[0]:
                cntyObj = obj
            else :
                continue

        #NORMALIZING MOBILITY

        normalize_mob1 = pd.concat([prenormalized_mob, cntyObj.get('MOBILITY TOTAL')], axis = 0)
        normalize_mob1 = normalize_mob1.to_numpy()
        normalize_mob1 = normalize_mob1.reshape(-1, 1)
        mob_scaled1 = min_max_scaler.fit_transform(normalize_mob1)
        mob_scaled1 = pd.DataFrame(mob_scaled1)
        mob_scaled1 = mob_scaled1.fillna(0)
        mob_scaled1 = mob_scaled1.tail(n=18)
        mob_scaled1.reset_index(drop=True, inplace=True)


        cntyObj = pd.DataFrame(cntyObj)
        cntyObj['MOBILITY TOTAL'] = mob_scaled1

        #cntyObj['PRECENT CASES BY AGE GROUP'].loc[(cntyObj['MOBILITY TOTAL'] > .7)] = cntyObj['PRECENT CASES BY AGE GROUP'] / .9

        ids = cntyObj[['CNTY NAME', 'AGE RANGE', 'PRECENT CASES BY AGE GROUP', 'MOBILITY TOTAL']]
        unique_data = cntyObj.drop(['CNTY NAME', 'AGE RANGE', 'IsOutbreak'], axis=1)
        user_results = pd.DataFrame(model.predict_proba(unique_data), columns=['Prob Not Outbreak', 'Predicted Prob Outbreak'])
        final_df = ids.join(user_results)

        final_df.columns = ['Location','Age Range', 'Percent of Cases By Age Group','Mobility Score', 'Predicted Probability of Not Outbreak', 'Predicted Probability of Outbreak']

        #Using this list to determine what age range user is in, to display specific age range
        ageList = [(0,4), (5,9), (10,14), (15,19), (20,24), (25,29), (30, 34), (35,39), (40, 44), (45,49), (50,54),(55,59), (60,64), (65,69), (70,74), (75,79), (80,84), (85, 999)]

        indexOfAge = 0

        for index, i in enumerate(ageList):
            if i[0] <= age and age <= i[1]:
                indexOfAge = index

        ageCol = final_df.iloc[indexOfAge]

        return (final_df, ageCol)

    except :
        return "error"



##########################################################
###################User Input#############################
##########################################################


############Change Info Here#####################
county = 'Los Angeles'
state = 'California'
age = 2
###############################

# validation and error and input parsing
county = county.lower()
state = state.lower()

# remove county from input
if 'county' in county: 
    county = county.replace('county', '')

# make first letter in words capital and remove whitespace
county = county.title().strip()
state = state.title().strip()

# get prediction
output = userInput(12, county, state)

# validate output and display
if len(output) == 2:
    print()
    print('||||||||||||||||||Results||||||||||||||||||')
    print('---------------------------------------')
    print()
    print('||||||||||Your Age Group||||||||||')
    print(output[1])
    print('---------------------------------------')
    print()

    print('||||||||||Your County||||||||||')
    print(output[0])
    print('---------------------------------------')
else:
    print()

    print('error getting your results, check inputs and try again')


##############################################################
