import sys
import pandas as pd
import numpy as np
import scipy.stats as stats


OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value: {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value: {more_searches_p:.3g}\n'
    '"Did more/less instructors use the search feature?" p-value: {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value: {more_instr_searches_p:.3g}'
)


def main():
    searchdata_file = sys.argv[1]
    searchData = pd.read_json(searchdata_file, orient='records', lines=True)
    #print(searchData.head())
    
    evenID_control = searchData[searchData['uid'] % 2 == 0]
    noSearch_control = evenID_control[evenID_control['search_count'] == 0]
    search_control = evenID_control[evenID_control['search_count'] > 0]
    
    oddID_treatment = searchData[searchData['uid'] % 2 != 0]
    noSearch_treatment = oddID_treatment[oddID_treatment['search_count'] == 0]
    search_treatment = oddID_treatment[oddID_treatment['search_count'] > 0]
    
    contingency = [[len(search_control), len(noSearch_control)], [len(search_treatment), len(noSearch_treatment)]]
    
    chi2, more_users_p, dof, expected = stats.chi2_contingency(contingency)                     #chi square test on all users

    statistic, more_searches_p = stats.mannwhitneyu(evenID_control['search_count'], oddID_treatment['search_count'])        #mann whitney test on all users


#--------------------------------------Repeated analysis looking only at instructors-------------------------------------------


    instructors = searchData[searchData['is_instructor'] == True]
    
    instruct_control = instructors[instructors['uid'] % 2 == 0]
    instruct_noSearch_control = instruct_control[instruct_control['search_count'] == 0]
    instruct_search_control = instruct_control[instruct_control['search_count'] > 0]
    
    instruct_treatment = instructors[instructors['uid'] % 2 != 0]
    instruct_noSearch_treatment = instruct_treatment[instruct_treatment['search_count'] == 0]
    instruct_search_treatment = instruct_treatment[instruct_treatment['search_count'] > 0]
    
    contingency2 = [[len(instruct_search_control), len(instruct_noSearch_control)], [len(instruct_search_treatment), len(instruct_noSearch_treatment)]]
    
    chi2, more_instr_p, dof, expected = stats.chi2_contingency(contingency2)                     #chi square test on only instructors

    statistic, more_instr_searches_p = stats.mannwhitneyu(instruct_control['search_count'], instruct_treatment['search_count'])        #mann whitney test on only instructors






    print(OUTPUT_TEMPLATE.format(
        more_users_p = more_users_p,
        more_searches_p = more_searches_p,
        more_instr_p = more_instr_p,
        more_instr_searches_p = more_instr_searches_p,
    ))

if __name__ == '__main__':
    main()
