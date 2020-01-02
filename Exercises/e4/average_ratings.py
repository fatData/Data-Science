import sys
import pandas as pd
import difflib as dl


movies = sys.argv[1]
ratings = sys.argv[2]
output = sys.argv[3]


movieData = open(movies).readlines()                             #get a list of lines
movieList = pd.DataFrame(data=movieData, columns=['title'])
movieList['title'] = movieList['title'].str.rstrip()            #removes trailing '\n' in each line
#print(movieList.head())

ratingsData = pd.read_csv(ratings)
#print(ratingsData.head())

titleOnly = ratingsData['title']
#print(titleOnly.head())


#Map function referenced from this link: 
#https://stackoverflow.com/questions/42698281/pandas-if-value-in-a-dataframe-contains-string-from-another-dataframe-append
correctedTitles = titleOnly.map(lambda x: dl.get_close_matches(x, movieList['title'], 1))
#print(correctedTitles.head())


#correctedTitles = correctedTitles.reset_index(drop=True)
ratingsData['title'] = correctedTitles.str[0]           #get rid of square brackets in list
#print(ratingsData.head())

ratingsData = ratingsData.join(movieList, on='title', how='outer', lsuffix='HasRating', rsuffix='NoRating')
#ratingsData.to_csv('testing.csv')           #checking how the data looks

ratingsData = ratingsData.drop(['title', 'titleNoRating'], axis= 1)     #drop those columns
#ratingsData.to_csv('testing2.csv')           #checking how the data looks

ratingsData = ratingsData.dropna()
#ratingsData.to_csv('testing3.csv')           #checking how the data looks

#grouped_data = ratingsData.groupby(['titleHasRating'])

avgRatings = ratingsData.groupby(['titleHasRating'])['rating'].mean()   #group by given column then take mean of corresponding values in other column
avgRatings = avgRatings.round(2)            #round to 2 sig figs
#avgRatings.columns = ['title','rating']                             

avgRatings.to_csv(output)           