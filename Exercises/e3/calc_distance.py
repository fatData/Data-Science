import sys
import pandas as pd
import numpy as np
from pykalman import KalmanFilter
from xml.dom.minidom import parse, getDOMImplementation
import math


def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')



#function below derived from:
#https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula?page=1&tab=votes#tab-top
def distance(theData):
    R = 6371       #Radius of the earth in km
    dLat = deg2rad(theData['lat'] - theData['lat0'])       #deg2rad below
    dLon = deg2rad(theData['lon'] - theData['lon0']) 
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(deg2rad(theData['lat0'])) * np.cos(deg2rad(theData['lat'])) * np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)); 
    theData = R * c       #Distance in km
    theData = theData * 1000            #converted to meters
    return theData.sum();


#function below derived from:
#https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula?page=1&tab=votes#tab-top
def deg2rad(deg):
    return deg * (math.pi/180)
    





doc_Object = parse(sys.argv[1])
df = pd.DataFrame(columns=['lat','lon'])
trkpt_elements = doc_Object.getElementsByTagName('trkpt')
    
for i in trkpt_elements:
    lat = i.getAttribute('lat')
    lon = i.getAttribute('lon')
    df = df.append({'lat': lat, 'lon': lon}, ignore_index=True)
        


shifted_data = df.shift(1)
shifted_data = shifted_data.rename(columns={'lat':'lat0', 'lon':'lon0'})
new_data = pd.concat([shifted_data, df], axis=1)

new_data = new_data.iloc[1:]        #makes dataframe start at index 1 so discards row at index 0

new_data = new_data.applymap(float)     #so that all data points rounded to same sigfigs

#print(new_data)


unfiltered_dist = distance(new_data)

print('Unfiltered distance: %0.2f' % unfiltered_dist)


#--------------------Kalman Filtering-------------------------

kalman_data = df.applymap(float)

initial_state = df.iloc[0]
#initial_state = pd.to_numeric(initial_state)

#print(initial_state)

st_d = 20/(10^5)
observation_covariance = np.diag([st_d, st_d]) ** 2

t_cov = 10/(10^5)                                   
transition_covariance = np.diag([t_cov, t_cov]) ** 2

transition = np.diag([1,1])

kf = KalmanFilter(initial_state_mean = initial_state,
                  observation_covariance = observation_covariance,
                  transition_covariance = transition_covariance,
                  transition_matrices = transition
                 )

pred_state, state_cov = kf.smooth(kalman_data)

filterData = pd.DataFrame(data=pred_state, columns=['lat', 'lon'])

filterData_shifted = filterData.shift(1)
filterData_shifted = filterData_shifted.rename(columns={'lat':'lat0', 'lon':'lon0'})

newFilterData = pd.concat([filterData_shifted, filterData], axis=1)
newFilterData = newFilterData.iloc[1:]

filtered_dist = distance(newFilterData)

print('Filtered distance: %0.2f' % filtered_dist)


output_gpx(filterData, 'out.gpx')
