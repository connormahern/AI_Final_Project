
path = '/Users/connormahern/Desktop/AI_Final_Project/'

censusData = path + 'AI_FinalProject/censusData.csv'
ageSexData = path + 'AI_FinalProject/ageSexData.csv'
#mobilityData = path + 'AI_FinalProject/google_mobility_data.csv'
popDensityData = path + 'AI_FinalProject/popDensityState.csv'
sexDistData = path + 'AI_FinalProject/output.csv'

import json
import urllib.request
import math
import pandas as pd
import heapq
import numpy as np
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from sklearn.linear_model import LinearRegression



url = "https://coronavirus.m.pipedream.net/"

# open json
with urllib.request.urlopen(url) as url:
    data = json.loads(url.read().decode())

# city object
class City():

    def __init__(self, place, state, country, cases, deaths, recovered, lat, long, unique):
        self.place = place
        self.state = state
        self.country = country
        self.cases = cases
        self.deaths = deaths
        self.recovered = recovered
        self.lat = lat
        self.long = long
        self.unique = unique

        if 0 <= int(self.cases) <= 100:
            self.severity = ["fcb900", 'low']
        elif 101 <= int(self.cases) <= 1000:
            self.severity = ["fc6100", 'medium']
        elif int(self.cases) > 1000:
            self.severity = ["ff0d00", 'high']


# put all data into this
caseInfo = data['rawData']

# init locations countries, and states
locations = []

# loop through data
for i in range(len(caseInfo)):

    # get current place
    place = caseInfo[i]

    # if there are any cases
    if int(place['Confirmed']) != 0:
        # create object  place, state, country, cases, deaths, recovered, lat, long, unique
        locations.append(
            City(place['Combined_Key'], place['Province_State'], place['Country_Region'], place['Confirmed'],
                  place['Deaths'], place['Recovered'], place['Lat'], place['Long_'], place['FIPS']))

# sort list by number of confirmed cases
locations.sort(key=lambda x: x.place, reverse=False)

"""## Find nearest city to user in corona dataset"""

def nearest(pointA):
    # function to calc distance
    def calc_euclidean_distance(point_a, point_b):
        return math.sqrt(math.pow(point_a[0] - point_b[0], 2) + math.pow(point_a[1] - point_b[1], 2))

    # init heap
    heap = []

    # loop through all locations to find closest
    for index, place in enumerate(locations):
        # if if its not empty
        if (place.long != "") and (place.lat != ""):

            long = float(place.long)
            lat = float(place.lat)

            pointB = [long, lat]
            d = [calc_euclidean_distance(pointA, pointB), index]

            heapq.heappush(heap, d)

    nearest = heapq.heappop(heap)[1]

    return(locations[nearest])

"""## Calc Probabilities"""

def prob(cityObj):

    # get string name
    strngName = cityObj.place.split(", ")

    mobilityData = ('https://raw.githubusercontent.com/ActiveConclusion/COVID19_mobility/master/google_reports/mobility_report_US.csv')
    dfMobility = pd.read_csv(mobilityData)
    dfMobility['date'] = pd.to_datetime(dfMobility['date'])
    dfMobility.fillna(0, inplace=True)
    try:
        df3 = dfMobility[dfMobility['state'].str.contains(locationName[1]) & dfMobility['county'].str.contains(locationName[0])]
        total = df3.tail(n=7)
        total = total.sum(axis=1, skipna=True)
        mobilityTotal = total.mean(axis=0)
        print('try', mobilityTotal)
    except:
        mobilityTotal = 0
        pass

    # read sex by county and returns a male to female distribution
    dfSDist = pd.read_csv(sexDistData)
    df4 = dfSDist[dfSDist['STNAME'].str.contains(strngName[1]) & dfSDist['CTYNAME'].str.contains(strngName[0])]

    #Adding all columns
    tot_pop = df4['TOT_POP']
    tot_pop = tot_pop[tot_pop.idxmax()]

    #MEAN DISTRIBUTION OF AGE BY COUNTY
    age_groups = ['Total', '0:4', '5:9',  '10:14', '15:19',  '20:24', '25:29', '30:34', '35:39', '40:44', '45:49', '50:54', '55:59', '60:64', '65:69', '70:74', '75:79', '80:84', '85+']
    age_tot = []

    for row in df4['TOT_POP'] :
        age_tot.append(row)

    tot_pop = age_tot[0]
    age_groups = age_groups[1:]
    age_tot = age_tot[1:]

    #CASES PER AGE RANGE
    casesT = cityObj.cases
    casesP = int(casesT) / tot_pop
    ageG_cases = []

    for row in age_tot :
        ageG_cases.append(math.floor(row * casesP))


    return {'CTNY' : cityObj.place, 'STN' : cityObj.state, 'TOT POP': tot_pop, 'TOT CASES': cityObj.cases, 'AGE RANGE' : age_groups, 'TPOP PA PCN' : age_tot, 'PRECENT OF POP INFECTED' : casesP, 'Number of Cases Per Age Range' : ageG_cases, 'MOBILITY TOTAL' : mobilityTotal}









