import pandas as pd

totals = pd.read_csv('totals.csv').set_index(keys=['name'])
counts = pd.read_csv('counts.csv').set_index(keys=['name'])

#print(totals)
#print(counts)

row_sum = totals.sum(axis=1)    #summation of percipitation values in each row

print('City with lowest total precipitation:\n' ,row_sum.idxmin(axis=0))       #return min index of the dataframe

col_precip = totals.sum(axis=0)     #summation of percipitation values in each column
#print(col_precip)
col_observ = counts.sum(axis=0)     #summation of observation values in each column
#print(col_observ)

print()
print('Average precipitation in each month:\n', col_precip/col_observ)

city_precip = totals.sum(axis=1)
#print(city_precip)
city_observ = counts.sum( axis=1)
#print(city_observ)

print()
print('Average precipitation in each city:\n', city_precip/city_observ)