import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

data = pd.read_csv('data.csv')

qs1 = data[data['algorithm'] == 'qs1'].reset_index(drop=None)
qs2 = data[data['algorithm'] == 'qs2'].reset_index(drop=None)
qs3 = data[data['algorithm'] == 'qs3'].reset_index(drop=None)
qs4 = data[data['algorithm'] == 'qs4'].reset_index(drop=None)
qs5 = data[data['algorithm'] == 'qs5'].reset_index(drop=None)
merge1 = data[data['algorithm'] == 'merge1'].reset_index(drop=None)
partition_sort = data[data['algorithm'] == 'partition_sort'].reset_index(drop=None)

anova = stats.f_oneway(merge1['elapsed_time'], partition_sort['elapsed_time'], qs1['elapsed_time'], qs2['elapsed_time'], qs3['elapsed_time'], qs4['elapsed_time'], qs5['elapsed_time'])

#print(anova)                #pvalue < .05 so there is difference between means of groups, so use Tukey test to identify which groups


melt_data = data[['algorithm', 'elapsed_time']]             #get required columns from data to use in the Tukey test
#print(melt_data)

posthoc = pairwise_tukeyhsd(melt_data['elapsed_time'], melt_data['algorithm'], alpha=0.05)
print(posthoc)

fig = posthoc.plot_simultaneous()

print(merge1['elapsed_time'].mean(), qs2['elapsed_time'].mean(), qs3['elapsed_time'].mean())                #print these means seperatley because the x-axis in the plot is not big enough

plt.show(fig)