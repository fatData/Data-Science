import sys
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
from pykalman import KalmanFilter


systemInfo = sys.argv[1]

table = pd.read_csv(systemInfo)
#print(table.head())

table['datetime']= pd.to_datetime(table['timestamp'])   #convert timestamp to datetime for plotting

plt.figure(figsize=(12, 4))
plt.plot(table['datetime'], table['temperature'], 'b.', alpha=0.5)
#plt.show()


filtered = lowess(table['temperature'], table['datetime'], frac=0.035)
#plt.figure(figsize=(12, 4))
plt.plot(table['datetime'], filtered[:, 1], 'r-', linewidth=3)
#plt.show()


kalman_data = table[['temperature', 'cpu_percent', 'sys_load_1']]       #important columns needed for kalman filtering

initial_state = kalman_data.iloc[0]         #row selection based on index

temp_dev = table['temperature'].std(axis=0)             #calc standard dev for observation covariance
cpu_percent_dev = table['cpu_percent'].std(axis=0)
sys_load_dev = table['sys_load_1'].std(axis=0)


observation_covariance = np.diag([temp_dev, cpu_percent_dev, sys_load_dev]) ** 2

transition_covariance = np.diag([.34, .34, .34]) ** 2

transition = [[1, -1, 0.7], [0, 0.6, 0.03], [0, 1.3, 0.8]]

kf = KalmanFilter(
    initial_state_mean=initial_state,
    observation_covariance=observation_covariance,
    transition_covariance=transition_covariance,
    transition_matrices=transition
    )

pred_state, state_cov = kf.smooth(kalman_data)

plt.plot(table['datetime'], pred_state[:, 0], 'g-', linewidth=3)     #kalman graph not plotting correct temperatures!!!!!!!!!!!!!!!!!!

plt.legend(['data points', 'LOESS-filter', 'Kalman-filter'] )
plt.savefig('cpu.svg')
#plt.show()