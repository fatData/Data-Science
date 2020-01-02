import numpy as np

data = np.load('monthdata.npz')
totals = data['totals']
counts = data['counts']

#print(totals)
#print(counts)


row_sum = np.sum(totals, axis=1)    #sum percipitation values in each row
#print(row_sum)

all_rows = np.reshape(row_sum, (9,1))   #reshape vector into a 9x1 matrix
#print(all_rows)

print('Row with lowest total precipitation: ', np.argmin(all_rows))     #determine row with lowest value


col_sum = np.sum(totals, axis=0)    #sum percipitation values in each column
#print(col_sum)

obs_col_sum = np.sum(counts, axis=0)    #sum observation values in each column
#print(obs_col_sum)

print()
print('Average precipitation in each month:\n ', np.divide(col_sum, obs_col_sum))     #divide two vectors


percip_rowSum = np.sum(totals, axis=1)
obs_rowSum = np.sum(counts, axis=1)

print()
print('Average precipitation in each city:\n ', np.divide(percip_rowSum, obs_rowSum))     #divide two vectors


n = totals.shape[0]     #get number of rows in matrix
#print(n)

percip_matrix = np.reshape(totals, (4*n,3))     #reshape matrix for easy quarterly calculations
#print(percip_matrix)

messy_percip_quarterly = np.sum(percip_matrix, axis=1)      #calc quarterly percip in each city
#print(percip_quarterly)

clean_quarterly_percip = np.reshape(messy_percip_quarterly, (n,4))      #reshape again for readability

print()
print('Quarterly precipitation totals:\n ', clean_quarterly_percip)