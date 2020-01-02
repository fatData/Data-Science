import sys
import gzip
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import scipy.stats as stats


OUTPUT_TEMPLATE = (
    "Initial (invalid) T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann–Whitney U-test p-value: {utest_p:.3g}"
)


def date_to_year(d):

	return '%04i' % (d.year)



def main():
    
    reddit_counts = sys.argv[1]
    reddit_counts_unzip = gzip.open(reddit_counts, 'rt')
    reddit_countsDF = pd.read_json(reddit_counts_unzip, lines=True)
    #print(reddit_countsDF.head())
    
    reddit_countsDF['year'] = reddit_countsDF['date'].apply(date_to_year)                   #seperate year from datetime object
    reddit_countsDF['year'] = pd.to_numeric(reddit_countsDF['year'])
    reddit_countsDF = reddit_countsDF[(reddit_countsDF['year'] == 2012) | (reddit_countsDF['year'] == 2013)]
    #print(reddit_countsDF.head())
    reddit_countsDF = reddit_countsDF[(reddit_countsDF['subreddit'] == 'canada')]
    reddit_countsDF = reddit_countsDF.sort_values(by=['date'])
    reddit_countsDF = reddit_countsDF.reset_index(drop=True)
    #print(reddit_countsDF.head())

    reddit_countsDF['day'] = reddit_countsDF['date'].apply(datetime.date.weekday)
    #print(reddit_countsDF.head())
    
    weekdays = reddit_countsDF[(reddit_countsDF['day'] != 5) & (reddit_countsDF['day'] != 6)]
    weekdays = weekdays.reset_index(drop=True)
    #print(weekdays.head())
    
    weekends = reddit_countsDF[(reddit_countsDF['day'] == 5) | (reddit_countsDF['day'] == 6)]
    weekends = weekends.reset_index(drop=True)
    #print(weekends.head())

    initial_ttest = stats.ttest_ind(weekdays['comment_count'], weekends['comment_count'])
    initial_ttest_p = initial_ttest.pvalue

    weekday_normaltest = stats.mstats.normaltest(weekdays['comment_count'])                 #check if data is normally distributed
    initial_weekday_normality_p = weekday_normaltest.pvalue
    
    weekend_normaltest = stats.mstats.normaltest(weekends['comment_count'])
    initial_weekend_normality_p = weekend_normaltest.pvalue

    initial_levene = stats.levene(weekdays['comment_count'], weekends['comment_count'], center='mean')                  #check if both data columns have equal variances
    initial_levene_p = initial_levene.pvalue

    #plt.hist(weekdays.comment_count)
    #plt.hist(weekends.comment_count)
    #plt.show()
    
    #weekdays['comment_count_log'] = np.log(weekdays['comment_count'])
    #testing = stats.mstats.normaltest(weekdays['comment_count_log'])
    #print(testing.pvalue)
    
    weekdays['comment_count_sqrt'] = np.sqrt(weekdays['comment_count'])                     #transforming weekday comments data column to get it closer to normality
    weekdays_normality_again = stats.mstats.normaltest(weekdays['comment_count_sqrt'])
    better_normality_weekdays = weekdays_normality_again.pvalue
    
    #weekdays['comment_count_sqr'] = weekdays['comment_count'] **2
    #testing = stats.mstats.normaltest(weekdays['comment_count_sqr'])
    #print(testing.pvalue)
    

    #weekends['comment_count_sqrt'] = np.sqrt(weekends['comment_count'])
    #weekend_normality_again = stats.mstats.normaltest(weekends['comment_count_sqrt'])
    #better_normality_weekends = weekend_normality_again.pvalue
    #print(weekend_normality_again.pvalue)
    
    weekends['comment_count_log'] = np.log(weekends['comment_count'])                       #transforming weekend comments data column to get it closer to normality
    weekend_normality_again = stats.mstats.normaltest(weekends['comment_count_log'])
    better_normality_weekends = weekend_normality_again.pvalue
    
    
    levene_again = stats.levene(weekdays['comment_count_sqrt'], weekends['comment_count_log'], center='mean')
    better_levene_p = levene_again.pvalue
    
    dateInfo = reddit_countsDF['date'].apply(datetime.date.isocalendar)             #used to calculate and extract week numbers from the 'date' column
    #print(dateInfo)
    
    year, weeknum, theday = zip(*dateInfo)          #split the info in the series
    year = list(year)                           #format info so it can be put into dataframe
    weeknum = list(weeknum)
    theday = list(theday)
    
    reddit_countsDF['diffYear'] = year
    reddit_countsDF['weekNum'] = weeknum
    reddit_countsDF['theday'] = theday
    
    
    reddit_countsDF = reddit_countsDF[(reddit_countsDF['diffYear'] == 2012) | (reddit_countsDF['diffYear'] == 2013)]
    
    weekdays2 = reddit_countsDF[(reddit_countsDF['theday'] != 6) & (reddit_countsDF['theday'] != 7)]        #filtering to get weekly weekday data
    weekdays2 = weekdays2.reset_index(drop=True)
    #print(weekdays2)
    
    weekdayDF = weekdays2.groupby(['diffYear', 'weekNum'])['comment_count'].mean()
    weekdayDF = weekdayDF.reset_index(drop=True)
    #print(weekdayDF)
    
    weekly_weekday = stats.mstats.normaltest(weekdayDF)                 #normality test for weekly weekday comments
    weekly_weekday_normality_p = weekly_weekday.pvalue
    
    
    
    
    
    weekends2 = reddit_countsDF[(reddit_countsDF['theday'] == 6) | (reddit_countsDF['theday'] == 7)]        #filtering to get weekly weekend data
    weekends2 = weekends2.reset_index(drop=True)
    #print(weekends2)
    
    weekndDF = weekends2.groupby(['diffYear', 'weekNum'])['comment_count'].mean()
    weekndDF = weekndDF.reset_index(drop=True)
    #print(weekndDF)
    
    weekly_weeknd = stats.mstats.normaltest(weekndDF)                   #normality test for weekly weekend comments
    weekly_weekend_normality_p = weekly_weeknd.pvalue
    
    
#    plt.hist(weekdayDF)
#    plt.hist(weekndDF)
#    plt.show()
#    print('the weekday mean: ', weekdayDF.mean())
#    print('the weekend mean: ', weekndDF.mean())
    
    
    anotha_levene = stats.levene(weekdayDF, weekndDF, center='mean')
    weekly_levene_p = anotha_levene.pvalue
    
    ttest_again= stats.ttest_ind(weekdayDF, weekndDF)
    again = ttest_again.pvalue
    
    utest = stats.mannwhitneyu(weekdays['comment_count'], weekends['comment_count'])
    utest_p = utest.pvalue
    



    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=initial_ttest_p,
        initial_weekday_normality_p=initial_weekday_normality_p,
        initial_weekend_normality_p=initial_weekend_normality_p,
        initial_levene_p=initial_levene_p,
        transformed_weekday_normality_p=better_normality_weekdays,
        transformed_weekend_normality_p=better_normality_weekends,
        transformed_levene_p=better_levene_p,
        weekly_weekday_normality_p=weekly_weekday_normality_p,
        weekly_weekend_normality_p=weekly_weekend_normality_p,
        weekly_levene_p=weekly_levene_p,
        weekly_ttest_p=again,
        utest_p=utest_p,
    ))


if __name__ == '__main__':
    main()