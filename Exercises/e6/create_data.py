import time
from implementations import all_implementations
import pandas as pd
import numpy as np



def which_sort(d):                                  #assign appropriate algorithm to each elapsed time 
    for i in range(0,len(d)):
        if i%7 == 0:
            d = d.set_value(i,'algorithm', 'qs1')
        if i%7 == 1:
            d = d.set_value(i,'algorithm', 'qs2')
        if i%7 == 2:
            d = d.set_value(i,'algorithm', 'qs3')
        if i%7 == 3:
            d = d.set_value(i,'algorithm', 'qs4')
        if i%7 == 4:
            d = d.set_value(i,'algorithm', 'qs5')
        if i%7 == 5:
            d = d.set_value(i,'algorithm', 'merge1')
        if i%7 == 6:
            d = d.set_value(i,'algorithm', 'partition_sort')
    return d

data = pd.DataFrame(columns=['start', 'finish', 'algorithm'])
start_data = pd.Series()
end_data = pd.Series()

for i in range(0,50):                                           #run each sorting algorithm 50 times
    random_array = np.random.randint(1000000,size=10000)
    for sort in all_implementations:
        st = time.time()
        res = sort(random_array)
        en = time.time()
        start_data = start_data.append(pd.Series(st))
        end_data = end_data.append(pd.Series(en))
        


start_data = start_data.reset_index(drop=True)
end_data= end_data.reset_index(drop=True)

data['start'] = start_data
data['finish'] = end_data
data['elapsed_time'] = data['finish'] - data['start']

data = which_sort(data)
data = data.sort_values('algorithm')


data.to_csv('data.csv', index=False)

