import sys
import pandas as pd
import matplotlib.pyplot as plt

filename1 = sys.argv[1]
filename2 = sys.argv[2]

table1 = pd.read_table(filename1, sep=' ', header=None, index_col=1,
        names=['lang', 'page', 'views', 'bytes'])

#print(table1.isnull().values.any())             #check for NaN values

table2 = pd.read_table(filename2, sep=' ', header=None, index_col=1,
        names=['lang', 'page', 'views', 'bytes'])


#print(table1.head())
#print(table2.head())

sort_by_views = table1.sort_values(by='views', ascending=False)
#print(sort_by_views.head())

plt.figure(figsize=(10, 5))     
plt.subplot(1, 2, 1)            # subplots in 1 row, 2 columns, select the first
plt.plot(sort_by_views['views'].values)                      # build plot 1
plt.title("Popularity Distribution")
plt.xlabel("Rank")
plt.ylabel("Views")
#plt.show()


join_tables = table1.join(table2, how='outer', lsuffix='tab1', rsuffix='tab2')
#print(join_tables.head())

clean_table = join_tables.dropna(axis=0, how='any', subset=['viewstab2'])       #remove Nan values
#print(clean_table.head())

plt.subplot(1, 2, 2)
plt.scatter(clean_table['viewstab1'].values, clean_table['viewstab2'].values)
plt.xscale('log')
plt.yscale('log')
plt.title("Daily Correlation")
plt.xlabel("Day 1 views")
plt.ylabel("Day 2 views")
plt.savefig("wikipedia.png")
#plt.show()
