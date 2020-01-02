import sys
import pandas as pd
import numpy as np
import gzip
import matplotlib.pyplot as plt
import math



#function below derived from:
#https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula?page=1&tab=votes#tab-top
def distance(city, station):
    R = 6371       #Radius of the earth in km
    dLat = deg2rad(city['latitude'] - station['latitude'])       #deg2rad below
    dLon = deg2rad(city['longitude'] - station['longitude']) 
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(deg2rad(station['latitude'])) * np.cos(deg2rad(city['latitude'])) * np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)); 
    dist = R * c       #Distance in km
    return dist


#function below derived from:
#https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula?page=1&tab=votes#tab-top
def deg2rad(deg):
    return deg * (math.pi/180)





def best_tmax(city, station):
    minDist = city.loc[city.groupby(['name'])['city_station_distance'].idxmin()]    #returns the min distance from station for each city
    minDist['important'] = minDist.index
    station['important'] = station.index
    avgMaxTemp_Included = station.join(minDist, on='important', how='inner', rsuffix='_city')       #take the intersection of the 'important' indexes and join into one table so that table can include corresponding avgMaxTemp
    
    return avgMaxTemp_Included




stationData = sys.argv[1]
cityData = sys.argv[2]
output = sys.argv[3]

station_jsonData = gzip.open(stationData, 'rt')
stations = pd.read_json(station_jsonData, lines=True)
cities = pd.read_csv(cityData)
#print(cities.head())
#print(stations.head())

stations['avg_tmax'] = stations['avg_tmax']/10
#print(stations.head())

cities = cities.dropna(axis=0, subset=['population', 'area'])
cities = cities.reset_index(drop=True)
cities['area'] = cities['area'] / 1000000       #convert column from m^2 to km^2
cities = cities[cities['area'] <= 10000]        #exclude certain areas
#print(cities.head())

countCities = len(cities)
countStations = len(stations)

newCityData = pd.concat([cities]*countStations, ignore_index=True)       #adds that many more rows of city data to the end of 'cities' dataset
#print(newCityData)
newCityData = newCityData.sort_values(by="name")
newCityData = newCityData.reset_index(drop=True)
#print(newCityData)

newStationData = pd.concat([stations]*countCities, ignore_index=True)       #adds that many more rows of station data to the end of 'stations' dataset
#print(newStationData.head())

newCityData['city_station_distance'] = distance(newCityData, newStationData)
#print(newCityData)

combined_dataset = best_tmax(newCityData, newStationData)
#print(combined_dataset)

combined_dataset['pop_density'] = combined_dataset['population'] / combined_dataset['area']
#print(combined_dataset.head())

plt.plot(combined_dataset['avg_tmax'], combined_dataset['pop_density'],'b.')
plt.xlabel("Avg Max Temperature (\u00b0C)")
plt.ylabel("Population Density (people/km\u00b2)")
plt.title("Temperature vs Population Density")

plt.savefig(output)

#print(combined_dataset['pop_density'].corr(combined_dataset['avg_tmax']))       #correlation coeff to determine whether the 2 variables are related

