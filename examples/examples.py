import csv
import glob
import sys
import operator
import itertools
from operator import itemgetter
import json
import pandas as pd

def readATPMatches(dirname):
    #ATP level
    allFiles = glob.glob(dirname + "/atp_matches_" + "????.csv")
    matches = pd.DataFrame()
    list_ = list()
    for filen in allFiles:
        df = pd.read_csv(filen,
                         index_col=None,
                         header=0)
        list_.append(df)
    matches = pd.concat(list_)
    return matches

def readChall_QATPMatches(dirname):
    #reads Challenger lvl + ATP Q matches
    allFiles = glob.glob(dirname + "/atp_matches_qual_chall_" + "????.csv")
    matches = pd.DataFrame()
    list_ = list()
    for filen in allFiles:
        df = pd.read_csv(filen,
                         index_col=None,
                         header=0)
        list_.append(df)
    matches = pd.concat(list_)
    return matches

#find single matches based on country and round
def matchesPerCountryAndRound(matches):
    matches = matches[(matches['round']=='F') & (matches['winner_ioc'] == 'AUT') & (matches['loser_ioc'] == 'AUT')]
    matches = matches.sort(['tourney_date'], ascending=False)
    #print matches.to_string(columns=['tourney_name','tourney_date','winner_name', 'loser_name'])
    print matches[['tourney_name','tourney_date','winner_name', 'loser_name']].to_csv(sys.stdout,index=False)
    
#looking for LLs who got deepes int grand slam draws starting from R32
def bestLLinGrandSlams(matches):
    matches = matches[((matches['round']=='R32') | (matches['round']=='R16') | (matches['round']=='QF') | (matches['round']=='SF') | (matches['round']=='F')) & (matches['tourney_level'] == 'G') & (matches['loser_entry'] == 'LL')]
    matches = matches.sort(['tourney_date'], ascending=False)
    print matches[['tourney_name','tourney_date','round','winner_name','winner_entry', 'loser_name', 'loser_entry']].to_csv(sys.stdout,index=False)    
  
#find matches longer then 'minutes' with 'sets' number of played sets
def numberOfSetsLongerThan(matches,sets,minutes):
    matches['score'].astype('str')
    matches = matches[(matches['minutes'] > minutes) & (matches['score'].str.count('-') == sets)]
    matches = matches.sort(['minutes'], ascending=False)
    print matches[['minutes','score','tourney_name','tourney_date','round','winner_name', 'loser_name']].to_csv(sys.stdout,index=False)    
    
#get all head-to-heads of the player
def geth2hforplayer(matches,name):
    matches = matches[(matches['winner_name'] == name) | (matches['loser_name'] == name)]
    h2hs = {}
    for index, match in matches.iterrows():
        if (match['winner_name'] == name):
            if (match['loser_name'] not in h2hs):
                h2hs[match['loser_name']] = {}
                h2hs[match['loser_name']]['l'] = 0
                h2hs[match['loser_name']]['w'] = 1
            else:
                h2hs[match['loser_name']]['w'] = h2hs[match['loser_name']]['w']+1
        elif (match['loser_name'] == name):
            if (match['winner_name'] not in h2hs):
                h2hs[match['winner_name']] = {}
                h2hs[match['winner_name']]['w'] = 0
                h2hs[match['winner_name']]['l'] = 1
            else:
                h2hs[match['winner_name']]['l'] = h2hs[match['winner_name']]['l']+1

    #create list
    h2hlist = []
    for k, v in h2hs.items():
        h2hlist.append([k, v['w'],v['l']])
    #sort by wins and then by losses + print
    print sorted(h2hlist, key=itemgetter(1,2))

#find if LL had to play same player in Q3/Q2 and MD of same tournament
def findLLQmultipleMatchesAtSameTournament(atpmatches,qmatches):
    resultlist = list()
    tourney_group = atpmatches.groupby('tourney_id')
    for tname, tdf in tourney_group:
        found1=False
        found2=False
        #first_case finds where a LL won against a Q in a main draw (MD)
        first_case = tdf[(tdf['winner_entry'] == 'LL') & (tdf['loser_entry'] == 'Q')]
        #iterating over first_case matches
        for index, match in first_case.iterrows():
            #looking for Q-finals where the loser matches the name of a winner of first_case matches 
            first_case_results = qmatches[(qmatches['tourney_name'] == match['tourney_name']+ ' Q') & ((qmatches['round'] =='Q2') | (qmatches['round'] =='Q3')) & (match['winner_name'] == qmatches['loser_name']) & (match['loser_name'] == qmatches['winner_name'])]
            if (len(first_case_results.index) > 0):
                #if results were found, add the MD match to the result list
                resultlist.append(first_case[((first_case['winner_name'] == first_case_results['loser_name']))])
    
          
        #second_case finds where a LL lost against a Q in a main draw (MD)  
        second_case = tdf[(tdf['winner_entry'] == 'Q') & (tdf['loser_entry'] == 'LL')]
        for index, match in second_case.iterrows():
            #looking for Q-finals where the loser matches the name of a loser of second_case matches
            second_case_results = qmatches[(qmatches['tourney_name'] == match['tourney_name']+ ' Q') & ((qmatches['round'] =='Q2') | (qmatches['round'] =='Q3')) & (match['winner_name'] == qmatches['winner_name']) & (match['loser_name'] == qmatches['loser_name'])]
            if (len(second_case_results.index) > 0):
                #if results were found, add the MD match to the result list
                resultlist.append(second_case[(second_case['loser_name'] == second_case_results['loser_name'])])
    
           
    result = pd.concat(resultlist).sort(['tourney_date'], ascending=False)
    print result[['tourney_name','tourney_date','round','winner_name','winner_entry', 'loser_name','loser_entry']].to_csv(sys.stdout,index=False)


#reading ATP level matches. The argument defines the path to the match files.
#since the match files are in the parent directory we provide ".." as an argument
atpmatches = readATPMatches("..")
#reading Challenger + ATP Q matches
qmatches = readChall_QATPMatches("..")

#the following lines make use of methods defined above this file. just remove the hash to uncomment the line and use the method.
#matchesPerCountryAndRound(matches)
#findLLQmultipleMatchesAtSameTournament(atpmatches,qmatches)
#bestLLinGrandSlams(atpmatches)
#numberOfSetsLongerThan(atpmatches,2,130)
geth2hforplayer(atpmatches,"Roger Federer")