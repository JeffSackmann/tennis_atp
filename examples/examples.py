'''
    File name: examples.py
    Description: all sorts of tennis-stats examples based on Jeff's tennis data (see: https://github.com/JeffSackmann/tennis_atp)
    Comment: the approach to do the calculations may not always be the most pythonic way. please forgive :)
    Author: Beta2k
    Python Version: 3
'''
import csv
import pprint
import datetime
import glob
import sys
import operator
import itertools
import collections
from operator import itemgetter
from collections import OrderedDict
import json
import numpy as np
import pandas as pd
import math
from pandas.core.categorical import Categorical
from spyderlib.widgets.externalshell import namespacebrowser



#util functions
def parse(t):
    ret = []
    for ts in t:
        try:
            string = str(ts)
            tsdt = datetime.date(int(string[:4]), int(string[4:6]), int(string[6:]))
        except TypeError:
            tsdt = datetime.date(1900,1,1)
        ret.append(tsdt)
    return ret

def readATPMatches(dirname):
    """Reads ATP matches but does not parse time into datetime object"""
    allFiles = glob.glob(dirname + "/atp_matches_" + "????.csv")
    matches = pd.DataFrame()
    container = list()
    for filen in allFiles:
        df = pd.read_csv(filen,
                         index_col=None,
                         header=0)
        container.append(df)
    matches = pd.concat(container)
    return matches

def readATPMatchesParseTime(dirname):
    """Reads ATP matches and parses time into datetime object"""
    allFiles = glob.glob(dirname + "/atp_matches_" + "????.csv")
    matches = pd.DataFrame()
    container = list()
    for filen in allFiles:
        df = pd.read_csv(filen,
                         index_col=None,
                         header=0,
                         parse_dates=[5],
                         encoding = "ISO-8859-1",
                         date_parser=lambda t:parse(t))
        container.append(df)
    matches = pd.concat(container)
    return matches

def readFMatches(dirname):
    """Reads ITF future matches but does not parse time into datetime object"""
    allFiles = glob.glob(dirname + "/atp_matches_futures_" + "????.csv")
    matches = pd.DataFrame()
    container = list()
    for filen in allFiles:
        df = pd.read_csv(filen,
                         index_col=None,
                         header=0)
        container.append(df)
    matches = pd.concat(container)
    return matches

def readFMatchesParseTime(dirname):
    """Reads ITF future matches but does not parse time into datetime object"""
    allFiles = glob.glob(dirname + "/atp_matches_futures_" + "????.csv")
    matches = pd.DataFrame()
    container = list()
    for filen in allFiles:
        df = pd.read_csv(filen,
                         index_col=None,
                         header=0,
                         parse_dates=[5],
                         encoding = "ISO-8859-1",
                         date_parser=lambda t:parse(t))
        container.append(df)
    matches = pd.concat(container)
    return matches

def readChall_QATPMatchesParseTime(dirname):
    """reads Challenger level + ATP Q matches and parses time into datetime objects"""
    allFiles = glob.glob(dirname + "/atp_matches_qual_chall_" + "????.csv")
    matches = pd.DataFrame()
    container = list()
    for filen in allFiles:
        df = pd.read_csv(filen,
                         index_col=None,
                         header=0,
                         parse_dates=[5],
                         encoding = "ISO-8859-1",
                         date_parser=lambda t:parse(t))
        container.append(df)
    matches = pd.concat(container)
    return matches

def readChall_QATPMatches(dirname):
    """reads Challenger level + ATP Q matches but does not parse time into datetime objects"""
    allFiles = glob.glob(dirname + "/atp_matches_qual_chall_" + "????.csv")
    matches = pd.DataFrame()
    container = list()
    for filen in allFiles:
        df = pd.read_csv(filen,
                         index_col=None,
                         header=0)
        container.append(df)
    matches = pd.concat(container)
    return matches

def readAllRankings(dirname):
    """reads all ranking files"""
    allFiles = glob.glob(dirname + "/atp_rankings_" + "*.csv")
    #allFiles = ['..\\atp_rankings_00s.csv', '..\\atp_rankings_10s.csv']
    ranks = pd.DataFrame()
    container = list()
    for filen in allFiles:
        df = pd.read_csv(filen,
                         index_col=None,
                         header=None,
                         parse_dates=[0],
                         encoding = "ISO-8859-1",
                         date_parser=lambda t:parse(t))
        container.append(df)
    ranks = pd.concat(container)
    return ranks

def parse_date(td):
    """helper function to parse time"""
    resYear = float(td.days)/364.0                   # get the number of years including the the numbers after the dot
    resMonth = int((resYear - int(resYear))*364/30)  # get the number of months, by multiply the number after the dot by 364 and divide by 30.
    resYear = int(resYear)
    return str(resYear) + "y" + str(resMonth) + "m"

def yearmonthdiff(row):
    s = row['ranking_date']
    e = row['dob']
    return relativedelta.relativedelta(s,e)
    
def get_date_wins(in_group): 
    temp = atpmatches[(atpmatches['winner_name'] == in_group.name) & (atpmatches['round'] == 'F')]
    in_group["tournament_wins"] = in_group.apply(lambda x: len(temp[temp['tourney_date'] < x['ranking_date']]), axis=1) 
    return in_group
    
def getRankForPreviousMonday(tdate,playername):
    """utility function to calculate the rank of a player from the previous week"""
    global joinedrankingsdf
    print(tdate)
    print(playername)
    #some tournaments start on a sunday, so we change this to a monday in order to get the correct ranking later on (we only have rankings for mondays obviously)
    if (tdate.weekday() != 0):
        diff = 7 - tdate.weekday()
        tdate = tdate + datetime.timedelta(days = diff)
    for x in range(1, 3):
        prevmon = tdate - datetime.timedelta(days = 7*x)
        if (len(joinedrankingsdf[(joinedrankingsdf['date'] == prevmon)]) > 0):
            rank = joinedrankingsdf[((joinedrankingsdf['fullname'] == playername) & (joinedrankingsdf['date'] == prevmon))].iloc[[0]]['rank'].values[0]
            return rank
            break
        else:
            continue

#calculations            
def matchesPerCountryAndRound(matches):
    """find single matches based on country and round"""
    matches = matches[(matches['round']=='F') & (matches['winner_ioc'] == 'AUT') & (matches['loser_ioc'] == 'AUT')]
    matches = matches.sort(['tourney_date'], ascending=False)
    #print matches.to_string(columns=['tourney_name','tourney_date','winner_name', 'loser_name'])
    print(matches[['tourney_name','tourney_date','winner_name', 'loser_name']].to_csv(sys.stdout,index=False))
    
def bestLLinGrandSlams(matches):
    """looking for LLs who got deepes int grand slam draws starting from R32"""
    matches = matches[((matches['round']=='R32') | (matches['round']=='R16') | (matches['round']=='QF') | (matches['round']=='SF') | (matches['round']=='F')) & (matches['tourney_level'] == 'G') & (matches['loser_entry'] == 'LL')]
    matches = matches.sort(['tourney_date'], ascending=False)
    print(matches[['tourney_name','tourney_date','round','winner_name','winner_entry', 'loser_name', 'loser_entry']].to_csv(sys.stdout,index=False))    
  
def numberOfSetsLongerThan(matches,sets,minutes):
    """find matches longer than 'minutes' with 'sets' number of played sets"""
    matches['score'].astype('str')
    matches = matches[(matches['minutes'] > minutes) & (matches['score'].str.count('-') == sets)]
    matches = matches.sort(['minutes'], ascending=False)
    print(matches[['minutes','score','tourney_name','tourney_date','round','winner_name', 'loser_name']].to_csv(sys.stdout,index=False))    
    
def geth2hforplayer(matches,name):
    """get all head-to-heads of the player"""
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
    #filter by h2hs with more than 6 wins:
    #h2hlist = [i for i in h2hlist if i[1] > 6]
    if (len(h2hlist) == 0):
        return ''
    else:
        return sorted(h2hlist, key=itemgetter(1,2))
        #for h2h in h2hlist:
        #    print(name+';'+h2h[0]+';'+str(h2h[1])+';'+str(h2h[2]))
    
def getWinLossByPlayer(qmatches, activeplayers, active):
    """returns w/l by player - change tourney level here (S=futures, C=challenger, etc.). of course matches of 
    the queries tourney level need to be in the matches provided as an argument to this function"""
    
    retmatches = qmatches[qmatches['score'].str.contains('RET').fillna(False)]
    retmatches_group = retmatches.groupby('loser_name').size().order(ascending=False)
    
    matches = qmatches[(qmatches['tourney_level'] == 'S')]
    finals = qmatches[(qmatches['round'] == 'F')]
    semifinals = qmatches[(qmatches['round'] == 'SF')]
    
    titles_group = finals.groupby('winner_name').size()
    finals_group = semifinals.groupby('winner_name').size()
    w_group = matches.groupby('winner_name').size()
    l_group = matches.groupby('loser_name').size()
    
    scores = pd.DataFrame({'wins' : w_group, 'losses' : l_group}).fillna(0)
    scores[['wins', 'losses']] = scores[['wins', 'losses']].astype(int)

    scores = scores.reindex_axis(['wins','losses'], axis=1)
    
    if (active):
        MAX_RANK = 20
        activeplayerslist = [row[0] for row in activeplayers]
        scores = scores[(scores.index.isin(activeplayerslist[:MAX_RANK]))]
        scores.index = pd.CategoricalIndex(scores.index, categories=activeplayerslist, ordered=True)
        #toggle sorting by ranking in next line
        scores = scores.sort_index()


    #todo: add titles and finals
    
    scores["matches"] = scores["wins"] + scores["losses"]
    scores["percentage"] = np.round(scores["wins"]*100/scores["matches"],2)
    #to see a column name when printig results
    scores.index.name = 'pname'
    scores = scores.join(pd.DataFrame(finals_group, columns = ['finals'],)).fillna(0)
    scores = scores.join(pd.DataFrame(titles_group, columns = ['titles'],)).fillna(0)
    scores = scores.join(pd.DataFrame(retmatches_group, columns = ['rets'],)).fillna(0)
    #changing datatype to int because it automatically is changed to float because of NaN values added for players which have 0 titles. 
    #even though the NaNs are immediately replaced by 0 values, the dtype of the column is changed from int to float. 
    #this is the reason why i need to change it back to int in the next line
    scores['titles'] = scores['titles'].astype('int')
    scores['finals'] = scores['finals'].astype('int')
    scores['rets'] = scores['rets'].astype('int')  
    
    #sort by wins
    scores = scores.sort(['titles'], ascending=False)
    print(scores.to_csv(sys.stdout,index=True))
    
def getRets(matches):
    matches = matches[matches['score'].str.contains('RET').fillna(False)]
    l_group = matches.groupby('loser_name').size().order(ascending=False)
    print(l_group.to_csv(sys.stdout,index=True))
    
def findLLQmultipleMatchesAtSameTournament(atpmatches,qmatches):
    """find if LL had to play same player in Q3/Q2 and MD of same tournament"""
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
    print(result[['tourney_name','tourney_date','round','winner_name','winner_entry', 'loser_name','loser_entry']].to_csv(sys.stdout,index=False))
    
def getActivePlayers(dirname):
    """finds active players, i.e. players who are in the current ranking"""
    currentRanking = dirname + "/atp_rankings_current.csv"
    playersDB = dirname + "/atp_players.csv"
    
    rankingdf = pd.DataFrame()
    playersdf = pd.DataFrame()
    
    rankingdf = pd.read_csv(currentRanking,index_col=None,header=None)
    rankingdf.columns = ['date', 'rank', 'id','points']
    playersdf = pd.read_csv(playersDB,index_col=None,header=None, encoding = "ISO-8859-1")
    playersdf.columns = ['id', 'fname', 'lname','hand','dob','country']
    maxdate = rankingdf['date'].max()
    rankingdf = rankingdf[(rankingdf['date'] == maxdate)]

    
    join = pd.merge(rankingdf,playersdf, on='id')
    join["fullname"] = join["fname"] + ' ' + join["lname"]
    join = join[['fullname','rank']]
    namelist = join.values.tolist()
    return namelist

def seedRanking(matches):
    """finds rankings of seeds"""
    #todo: create new DF to merge when seed won/lost into one column.
    #also take into account that in old draws the seeds did not have byes in R32. this also needs to be filtered.
    wmatches = matches[((matches['round'] == 'R16') & (matches['winner_seed'] == 3) & (matches['winner_rank'] > 40))]
    print(wmatches[['tourney_name','tourney_date','winner_name', 'winner_rank', 'winner_seed']].to_csv(sys.stdout,index=False))
    lmatches = matches[(matches['round'] == 'R16') & (matches['loser_seed'] == 3) & (matches['loser_rank'] > 40)]
    print(lmatches[['tourney_name','tourney_date','loser_name', 'loser_rank', 'loser_seed']].to_csv(sys.stdout,index=False))
    
def qualifierSeeded(atpmatches):
    """finds qualifiers, which were seeded"""
    lmatches = atpmatches[((atpmatches['loser_entry'] == 'Q') & (atpmatches['loser_seed'] < 9))]
    lmatches = lmatches.rename(columns={'loser_entry': 'entry', 'loser_seed': 'seed', 'loser_name': 'name'})
    wmatches = atpmatches[((atpmatches['winner_entry'] == 'Q') & (atpmatches['winner_seed'] < 9) & (atpmatches['round'] == 'F'))]
    wmatches = wmatches.rename(columns={'winner_entry': 'entry', 'winner_seed': 'seed', 'winner_name': 'name'})
    frames = [lmatches, wmatches]
    result = pd.concat(frames)
    result['seed'] = result['seed'].astype('int')
    result = result.sort(['tourney_date','tourney_name'], ascending=[True,True])
    print(result[['tourney_date','tourney_name','name','entry','seed','round']].to_csv(sys.stdout,index=False))
    
def getDictEightSeedRankperTourney(matches):
    """util function which returns a dictionary containing ranks of 8 seeds per tournament"""
    #max_seed for ATP = 32 + buffer
    #max_seed for CH = 8 + buffer
    MAX_SEED = 32
    tgroup = matches.groupby('tourney_id')
    rankdict = {}
    rankdict['id'] = {}
    rankdict['altid'] = {}
    for tid, t in tgroup:
        #print t
        found = False
        for n in range(MAX_SEED,6,-1):
            if (len(t[(t['loser_seed'] == n)]) > 0):
                tr = t[(t['loser_seed'] == n)]
                eightrank = tr.iloc[[0]]['loser_rank']
                eightrank = eightrank.values[0]
                found = True
            elif (len(t[(t['winner_seed'] == n)]) > 0):
                tr = t[(t['winner_seed'] == n)]
                eightrank = tr.iloc[[0]]['winner_seed']
                eightrank = eightrank.values[0]
                found = True
            if (found):
                #print 'added ' + str(tid)
                rankdict[tid] = eightrank
                tname = tr.iloc[[0]]['tourney_name'].values[0]
                tyear = tr.iloc[[0]]['tourney_date'].values[0]
                rankdict['id'][tid] = eightrank
                altid = str(tname)+str(tyear)[:4]
                #print 'added ' + str(altid) 
                rankdict['altid'][altid] = eightrank
                break
    return rankdict

def highRankedQLosers(qmatches,atpmatches):
    """finds high ranked losers of qualification draws"""
    amatches = atpmatches[((atpmatches['tourney_level'] == 'A') & (atpmatches['tourney_date'] > 20000000))]
    qmatches = qmatches[((qmatches['tourney_level'] == 'Q') & (qmatches['tourney_date'] > 20000000))]
            
    rankdict = getDictEightSeedRankperTourney(amatches)
    rankdict = rankdict['altid']
    rankdf = pd.DataFrame(list(rankdict.items()),columns=['id','8seedrank'])
    rankdf['year'] = rankdf.id.str[-4:]
    rankdf['year'] = rankdf['year'].astype(int) 
    rankdf['tourney_name'] = rankdf.id.str[:-4] + ' Q'
    qmatches['date4'] = (qmatches['tourney_date']/10000).astype(int)
    merged = rankdf.merge(qmatches, left_on=['year', 'tourney_name'], right_on=['date4', 'tourney_name'])
    merged = merged[(merged['loser_rank'] < merged['8seedrank'])]
    
    print(merged[['tourney_id','tourney_date','tourney_name','loser_name','loser_rank', '8seedrank']].sort(['tourney_date'], ascending=[True]).to_csv(sys.stdout,index=False))    
 
def fedR4WimbiTime(atpmatches):
    """shows the time federer spent on court until 4th rounds in wimbledon"""
    atpmatches = atpmatches[(atpmatches['tourney_name'] == 'Wimbledon')]
    atpmatches = atpmatches[(atpmatches['winner_name'] == 'Roger Federer')]
    atpmatches = atpmatches[(atpmatches['round'] == 'R128') | (atpmatches['round'] == 'R64') | (atpmatches['round'] == 'R32') | (atpmatches['round'] == 'R16')]
    matchesgroup = atpmatches.groupby('tourney_id')
    print(matchesgroup['minutes'].sum()) 

def youngFutures(matches):
    """finds young futures players
    set round and age parameter in next line"""
    matches = matches[(matches['round'] == 'QF') & (matches['winner_age'] < 16)]
    matches = matches.sort(['winner_age'], ascending=True)
    print(matches[['tourney_name','tourney_date','winner_name', 'winner_age', 'loser_name']].to_csv(sys.stdout,index=False))
    
def rankofQhigherthanlastSeed(matches):
    """find players and tournaments where the rank of the 1-seeded player in the qualies-draw
    was higher than the last seed of the main draw"""
    #if input is challenger use next line
    #matches = matches[((matches['tourney_level'] == 'C') & (matches['tourney_date'] > 20000000))]
    #if input is atp tour use next line
    matches = matches[((matches['tourney_level'] == 'A') & (matches['tourney_date'] > 20100000))]
            
    rankdict = getDictEightSeedRankperTourney(matches)
    rankdict = rankdict['id']
        
    results = {}
    matches = matches[((matches['winner_entry'] == 'Q') | (matches['loser_entry'] == 'Q'))]
    qgroup = matches.groupby('tourney_id')
    for tid, t in qgroup:
        #winner = q
        for index, match in t[(t['winner_entry'] == 'Q')].iterrows():
            try:
                if (match['winner_rank'] < rankdict[tid]):
                    if (tid in results):
                        if (match['winner_name'] not in results[tid]['players']):
                            results[tid]['players'].append(match['winner_name'])
                    else:
                        results[tid] = {}
                        results[tid]['id'] = match['tourney_id']
                        results[tid]['name'] = match['tourney_name']
                        results[tid]['date'] = match['tourney_date']
                        results[tid]['players'] = []
                        results[tid]['players'].append(match['winner_name'])
            except KeyError:
                continue

        #loser = q
        for index, match in t[(t['loser_entry'] == 'Q')].iterrows():
            try:
                if (match['loser_rank'] < rankdict[tid]):
                    if (tid in results):
                        if (match['loser_name'] not in results[tid]['players']):
                            results[tid]['players'].append(match['loser_name'])
                    else:
                        results[tid] = {}
                        results[tid]['id'] = match['tourney_id']
                        results[tid]['name'] = match['tourney_name']
                        results[tid]['date'] = match['tourney_date']
                        results[tid]['players'] = []
                        results[tid]['players'].append(match['loser_name'])
            except KeyError:
                continue
    
    orderedRes = (OrderedDict(sorted(results.items(), key=lambda x: x[1]['date'])))
    for t in orderedRes:
        playerstring = ','.join(orderedRes[t]['players'])
        
        yearid = t.split('-')
        year = yearid[0]
        id = yearid[1]
        md = 'http://www.atpworldtour.com/posting/'+str(year)+'/'+str(id)+'/mds.pdf'
        qd = 'http://www.atpworldtour.com/posting/'+str(year)+'/'+str(id)+'/qs.pdf'
        
        print(str(orderedRes[t]['date'])+','+orderedRes[t]['name']+',' + playerstring + ','+md+','+qd)
        
def avglastseedrank(matches):
    """calculates the average of the last seed rank per tournament category"""
    #only matches from 2013 and 2014
    matches = matches[(matches['tourney_date'] > datetime.date(2012,12,29)) & (matches['tourney_date'] < datetime.date(2015,1,1))]
    
    #atp 500
    #if draw size = 32, then 8 seeds
    #if draw size = 48, then 16 seeds
    #if draw size = 56, then 16 seeds
    tourney500names = ['Rotterdam', 'Rio de Janeiro', 'Acapulco', 'Dubai', 'Barcelona', 'Hamburg', 'Washington', 'Beijing', 'Tokyo', 'Valencia', 'Basel', 'Memphis']
    
    matches500 = matches[matches['tourney_name'].isin(tourney500names)]
    #remove 2014-402 (= memphis) because in 2014 it was a 250
    matches500 = matches500[(matches500['tourney_id'] != '2014-402')]
    matches500group = matches500.groupby('tourney_id')
    print("500============")
    getLastSeedRankForGroupedTourneys(matches500group)

    
    #atp 1000
    #if draw size = 48, then 16 seeds
    #if draw size = 56, then 16 seeds
    #if draw size = 96, then 32 seeds
    matches1000 = matches[(matches['tourney_level'] == 'M')]
    matches1000group = matches1000.groupby('tourney_id')
    print("1000============")
    getLastSeedRankForGroupedTourneys(matches1000group)
    
    #atp 250
    #if draw size = 28, then 8 seeds
    #if draw size = 32, then 8 seeds
    #if draw size = 48, then 16 seeds
    memphis2014 = matches[(matches['tourney_id'] == '2014-402')]
    matches250 = matches[(matches['tourney_level'] == 'A')]
    matches250 = matches250[~matches250['tourney_name'].isin(tourney500names)]
    #add memphis for 2014, because it became a 250 
    matches250 = pd.concat([matches250, memphis2014])
    matches250group = matches250.groupby('tourney_id')
    print("250============")
    getLastSeedRankForGroupedTourneys(matches250group)
    
    #Grand Slam
    #if draw size = 128, then 32 seeds
    matchesG = matches[(matches['tourney_level'] == 'G')]
    matchesGgroup = matchesG.groupby('tourney_id')
    print("GS============")
    getLastSeedRankForGroupedTourneys(matchesGgroup)

def rankingPointsOfYoungsters(players,ranks):
    """calculates ranking points of young players to answer questions like 
    "how many ranking points did players younger than 16y have?"""
    players.columns = ['id', 'fname', 'lname', 'hand', 'dob', 'country']
    players['full_name'] = players['fname'] + ' ' + players['lname']
    ranks.columns = ['ranking_date', 'rank', 'player_id', 'points']
    join = pd.merge(ranks,players, left_on='player_id', right_on='id')
    join['age'] = join['ranking_date'] - join['dob']
    join = join[join['age'] > datetime.timedelta(days = 0)]
    join = join[(join['age'] < datetime.timedelta(days = 5875))]
    #join['readAge'] = [parse_date(ranking_date - dob) for ranking_date, dob in zip(join["ranking_date"], join["dob"])]
    join['readAge'] = join.apply(yearmonthdiff, axis=1)
    join['points'] = join['points'].astype('int')
    join = join[(join['points'] > 5)]
    join = join.groupby('full_name', as_index=False).apply(lambda g: g.loc[g.age.idxmin()])
    join = join.sort(['age'], ascending=False)
    print(join[['ranking_date','dob', 'points','rank','readAge',  'full_name', 'country']].to_csv(sys.stdout,index=False))

def getLastSeedRankForGroupedTourneysDeprecated(groupedmatches):
    """DEPRECATED: returns the rank of the last seed for a give tournament"""
    resultlist = []
    resultlist8 = []
    resultlist16 = []
    resultlist32 = []
    for tid, tmatches in groupedmatches:
        print(tid + ' - ' + str(len(tmatches)) + ' - ' + tmatches.iloc[[0]]['tourney_name'].values[0])
        if ((len(tmatches) == 55) | (len(tmatches) == 47)):
            maxseed = 16
            #look for ranking of 16 seed
        elif ((len(tmatches) == 31) | (len(tmatches) == 27)):
            #look for ranking of 8 seed
            maxseed = 8
        elif ((len(tmatches) == 95) | (len(tmatches) == 127)) :
            #look for ranking of 32 seed
            maxseed = 32
        try: 
            tempmatches = tmatches[(tmatches['loser_seed'] == maxseed)]
            if (len(tempmatches) == 1):
                rank = tempmatches.iloc[[0]]['loser_rank'].values[0]
            elif (len(tempmatches) < 1):
                tempmatches = tmatches[(tmatches['winner_seed'] == maxseed)]
                rank = tempmatches.iloc[[0]]['winner_rank'].values[0]
            print(rank)
            resultlist.append(rank)
            if (maxseed == 8):
                resultlist8.append(rank)
            elif (maxseed == 16):
                resultlist16.append(rank)
            elif (maxseed == 32):
                resultlist32.append(rank)            
        except:
            continue

    print("Overall:")
    print(resultlist)
    resultlist = np.asarray(resultlist)
    print("Mean : {0:8.2f}".format(resultlist.mean()))
    print("Minimum : {0:8.0f}".format(resultlist.min()))
    print("Maximum : {0:8.0f}".format(resultlist.max()))
    print("Std. deviation : {0:8.2f}".format(resultlist.std()))
    
    if (len(resultlist8) > 0):
        print("8 Seeds:")
        print(resultlist8)
        resultlist8 = np.asarray(resultlist8)
        print("Mean : {0:8.2f}".format(resultlist8.mean()))
        print("Minimum : {0:8.0f}".format(resultlist8.min()))
        print("Maximum : {0:8.0f}".format(resultlist8.max()))
        print("Std. deviation : {0:8.2f}".format(resultlist8.std()))   
        
    if (len(resultlist16) > 0):
        print(resultlist16)
        resultlist16 = np.asarray(resultlist16)
        print("16 Seeds:")
        print("Mean : {0:8.2f}".format(resultlist16.mean()))
        print("Minimum : {0:8.0f}".format(resultlist16.min()))
        print("Maximum : {0:8.0f}".format(resultlist16.max()))
        print("Std. deviation : {0:8.2f}".format(resultlist16.std()))   
        
    if (len(resultlist32) > 0):
        print("32 Seeds:")
        print(resultlist32)
        resultlist32 = np.asarray(resultlist32)
        print("Mean : {0:8.2f}".format(resultlist32.mean()))
        print("Minimum : {0:8.0f}".format(resultlist32.min()))
        print("Maximum : {0:8.0f}".format(resultlist32.max()))
        print("Std. deviation : {0:8.2f}".format(resultlist32.std()))   
        
def getLastSeedRankForGroupedTourneys(groupedmatches):
    """returns the rank of the last seed for a give tournament"""
    global joinedrankingsdf
    #read rankings
    dirname = ".."
    ranking10s = dirname + "/atp_rankings_10s.csv"
    playersDB = dirname + "/atp_players.csv"
    
    rankingdf = pd.DataFrame()
    playersdf = pd.DataFrame()
    
    rankingdf = pd.read_csv(ranking10s,index_col=None,header=None,
                         parse_dates=[0],
                         date_parser=lambda t:parse(t))
    rankingdf.columns = ['date', 'rank', 'id','points']
    playersdf = pd.read_csv(playersDB,index_col=None,header=None)
    playersdf.columns = ['id', 'fname', 'lname','hand','dob','country']
    
    joinedrankingsdf = pd.merge(rankingdf,playersdf, on='id')
    joinedrankingsdf["fullname"] = joinedrankingsdf["fname"] + ' ' + joinedrankingsdf["lname"]

    resultlist = []
    resultlist8 = []
    resultlist16 = []
    resultlist32 = []
    for tid, tmatches in groupedmatches:
        #if (tid == '2013-404'):
        print(tid + ' - ' + str(len(tmatches)) + ' - ' + tmatches.iloc[[0]]['tourney_name'].values[0])
        if ((len(tmatches) == 55) | (len(tmatches) == 47)):
            maxseed = 16
            #look for ranking of 16 seed
        elif ((len(tmatches) == 31) | (len(tmatches) == 27)):
            #look for ranking of 8 seed
            maxseed = 8
        elif ((len(tmatches) == 95) | (len(tmatches) == 127)) :
            #look for ranking of 32 seed
            maxseed = 32
        try: 
            tempmatches = tmatches[(tmatches['loser_seed'] == maxseed)]
            if (len(tempmatches) == 1):
                #rank = tempmatches.iloc[[0]]['loser_rank'].values[0]
                playername = tempmatches.iloc[[0]]['loser_name'].values[0]
                tdate = tempmatches.iloc[[0]]['tourney_date'].values[0]
                #try previous mondays and if found we are fine.
                rank = getRankForPreviousMonday(tdate,playername)
            elif (len(tempmatches) < 1):
                tempmatches = tmatches[(tmatches['winner_seed'] == maxseed)]
                #rank = tempmatches.iloc[[0]]['winner_rank'].values[0]
                playername = tempmatches.iloc[[0]]['winner_name'].values[0]
                tdate = tempmatches.iloc[[0]]['tourney_date'].values[0]
                #try previous mondays
                rank = getRankForPreviousMonday(tdate,playername)
            print(rank)
            resultlist.append(rank)
            if (maxseed == 8):
                resultlist8.append(rank)
            elif (maxseed == 16):
                resultlist16.append(rank)
            elif (maxseed == 32):
                resultlist32.append(rank)        
        except Exception as e:
            s = str(e)
            print(e)
            print("Exception likely due to last seed having withdrawn. So we ignore this and it's fine!")
            continue

    print("Overall:")
    print(resultlist)
    resultlist = np.asarray(resultlist)
    print("Mean : {0:8.2f}".format(resultlist.mean()))
    print("Minimum : {0:8.0f}".format(resultlist.min()))
    print("Maximum : {0:8.0f}".format(resultlist.max()))
    print("Std. deviation : {0:8.2f}".format(resultlist.std()))
    
    if (len(resultlist8) > 0):
        print("8 Seeds:")
        print(resultlist8)
        resultlist8 = np.asarray(resultlist8)
        print("Mean : {0:8.2f}".format(resultlist8.mean()))
        print("Minimum : {0:8.0f}".format(resultlist8.min()))
        print("Maximum : {0:8.0f}".format(resultlist8.max()))
        print("Std. deviation : {0:8.2f}".format(resultlist8.std()))   
        
    if (len(resultlist16) > 0):
        print(resultlist16)
        resultlist16 = np.asarray(resultlist16)
        print("16 Seeds:")
        print("Mean : {0:8.2f}".format(resultlist16.mean()))
        print("Minimum : {0:8.0f}".format(resultlist16.min()))
        print("Maximum : {0:8.0f}".format(resultlist16.max()))
        print("Std. deviation : {0:8.2f}".format(resultlist16.std()))   
        
    if (len(resultlist32) > 0):
        print("32 Seeds:")
        print(resultlist32)
        resultlist32 = np.asarray(resultlist32)
        print("Mean : {0:8.2f}".format(resultlist32.mean()))
        print("Minimum : {0:8.0f}".format(resultlist32.min()))
        print("Maximum : {0:8.0f}".format(resultlist32.max()))
        print("Std. deviation : {0:8.2f}".format(resultlist32.std()))
        
def getBestQGrandSlamPlayer(qmatches,rankings):
    """returns highgest ranked players in grand slame quali-draws in order to find out the best cutoff for grand slam qualies"""
    global joinedrankingsdf
    #join rankings with playernames
    dirname = ".."
    playersDB = dirname + "/atp_players.csv"
    
    rankings.columns = ['date', 'rank', 'id','points']
    playersdf = pd.read_csv(playersDB,index_col=None,header=None)
    playersdf.columns = ['id', 'fname', 'lname','hand','dob','country']
    
    joinedrankingsdf = pd.merge(rankings,playersdf, on='id')
    joinedrankingsdf["fullname"] = joinedrankingsdf["fname"] + ' ' + joinedrankingsdf["lname"]
   
    qmatches = qmatches[(qmatches['tourney_name'] == 'Australian Open Q') | (qmatches['tourney_name'] == 'Roland Garros Q') | (qmatches['tourney_name'] == 'US Open Q') | (qmatches['tourney_name'] == 'Wimbledon Q')]
    matchesgroup = qmatches.groupby('tourney_id')
    res = {}
    for tid, tmatches in matchesgroup:
        name = tid[:4] + '-'  + tmatches.iloc[[0]]['tourney_name'].values[0]
        print(name)
        
        #get all players into a set
        w_set = set(tmatches['winner_name'])
        l_set = set(tmatches['loser_name'])
        #u_set contains all names of participating players
        u_set = w_set.union(l_set)      
        #get ranking date
        tdate = tmatches.iloc[[0]]['tourney_date'].values[0]
        #q deadline is 4 weeks earlier
        #deadline_date = tdate -  datetime.timedelta(days = 28)
        #alternatively take 6 weeks earlier deadline (= md deadline)
        deadline_date = tdate -  datetime.timedelta(days = 42)
        
        if (deadline_date.weekday() == 6):
            deadline_date = deadline_date + datetime.timedelta(days = 1)
        
        #get rankings for each player in the set for deadline_date
        player_list = list(u_set)
        plist_df = pd.DataFrame(player_list)
        plist_df.columns = ['fullname']
        plist_df['entry_date'] = deadline_date
        merged = plist_df.merge(joinedrankingsdf, left_on=['fullname', 'entry_date'], right_on=['fullname', 'date'])
        merged = merged.sort(['rank'], ascending=True)
        #print(merged[['fullname', 'rank']].head(1))
        print(merged[['fullname', 'rank']].head(1).to_csv(sys.stdout, header=False, index=False))
        fullname = merged.head(1).iloc[[0]]['fullname'].values[0]
        rank = merged.head(1).iloc[[0]]['rank'].values[0]
        res[name] = [fullname , rank]
        
    for key, value in sorted(res.items(), key=lambda e: e[1][1]):
        print(key+','+value[0]+','+str(value[1]))
       
def getAces(matches):
    """find matches where a player hit many aces.
    define the threshold in the next line"""
    matches = matches[((matches['w_ace'] > 45) | (matches['l_ace'] > 45))]
    print(matches[['tourney_date','tourney_name','winner_name','loser_name','w_ace','l_ace', 'score']].to_csv(sys.stdout,index=False))
    
def getShortestFiveSetter(matches):
    """finds short 5 set matches.
    define your own thresholds by changing the values in the line after the next."""
    matches['score'].astype('str')
    matches = matches[(matches['minutes'] < 150) & (matches['score'].str.count('-') == 5)]
    matches = matches.sort(['minutes'], ascending=True)
    print(matches[['minutes','score','tourney_name','tourney_date','round','winner_name', 'loser_name']].to_csv(sys.stdout,index=False))   
    
def getworstlda(matches):
    """find the worst tournaments in terms of 'last direct acceptance'"""
    global joinedrankingsdf
    
    #join rankings with playernames
    dirname = ".."
    playersDB = dirname + "/atp_players.csv"
    
    rankings.columns = ['date', 'rank', 'id','points']
    playersdf = pd.read_csv(playersDB,index_col=None,header=None)
    playersdf.columns = ['id', 'fname', 'lname','hand','dob','country']
    
    joinedrankingsdf = pd.merge(rankings,playersdf, on='id')
    joinedrankingsdf["fullname"] = joinedrankingsdf["fname"] + ' ' + joinedrankingsdf["lname"]
    
    matches = matches[((matches['tourney_level'] == 'A') & (matches['tourney_date'] > datetime.date(2007,1,1)) & ((matches['draw_size'] == 32) | (matches['draw_size'] == 28)))]
    
    tgroup = matches.groupby('tourney_id')
    res = {}
    for tid, tmatches in tgroup:
        name = tid + '-'  + tmatches.iloc[[0]]['tourney_name'].values[0]
        print(name)
        #get all players into a set
        w_set = set(tmatches[(tmatches['winner_entry'] != 'Q') & (tmatches['winner_entry'] != 'WC') & (tmatches['winner_entry'] != 'LL')]['winner_name'])
        l_set = set(tmatches[(tmatches['loser_entry'] != 'Q') & (tmatches['loser_entry'] != 'WC') & (tmatches['loser_entry'] != 'LL')]['loser_name'])
        #u_set contains all names of participating players
        u_set = w_set.union(l_set)
        #get ranking date
        tdate = tmatches.iloc[[0]]['tourney_date'].values[0]
        deadline_date = tdate -  datetime.timedelta(days = 42)
        #print(deadline_date.weekday())
        if (deadline_date.weekday() == 6):
            deadline_date = deadline_date + datetime.timedelta(days = 1)
            
        print(deadline_date)
        for x in range(0, 3):
            deadline_date = deadline_date - datetime.timedelta(days = 7*x)
            if (len(joinedrankingsdf[(joinedrankingsdf['date'] == deadline_date)]) > 0):
                print("gefunden")
                print(deadline_date)
                break
            else:
                continue
            
        #get rankings for each player in the set for deadline_date
        player_list = list(u_set)
        plist_df = pd.DataFrame(player_list)
        plist_df.columns = ['fullname']
        plist_df['entry_date'] = deadline_date
        merged = plist_df.merge(joinedrankingsdf, left_on=['fullname', 'entry_date'], right_on=['fullname', 'date'])
        merged = merged.sort(['rank'], ascending=False)
        print(merged[['fullname', 'rank']].head(1).to_csv(sys.stdout, header=False, index=False))
        try: 
            fullname = merged.head(1).iloc[[0]]['fullname'].values[0]
            rank = merged.head(1).iloc[[0]]['rank'].values[0]
            res[name] = [fullname , rank]
        except IndexError:
            continue
            
        
    for key, value in sorted(res.items(), key=lambda e: e[1][1]):
        print(key+','+value[0]+','+str(value[1]))

def getCountriesPerTournament(matches):
    """returns tournaments where many players of the same country participated.
    currently the function operates on challenger matches.
    parameters in the next line may be changed to do the same for ATP tour level matches (no guarantee that it works without any further modifications)"""
    matches = matches[(matches['tourney_level'] == 'C') & (matches['round'] == 'R32')]
    tgroup = matches.groupby('tourney_id')
    res = {}
    for tid, tmatches in tgroup:
        name = tid + '-'  + tmatches.iloc[[0]]['tourney_name'].values[0]
        #print(name)
        
        #get all winner countries into a set
        w_list = list(tmatches['winner_ioc'])
        l_list = list(tmatches['loser_ioc'])
        u_list = w_list + l_list
        top = collections.Counter(u_list).most_common(1)
        res[name] = [top[0][0], top[0][1]]
        #print(res)
        
        #u_set contains all names of participating players
        #u_set = w_set.union(l_set)
        
    for key, value in sorted(res.items(), key=lambda e: e[1][1], reverse=True):
        print(key+','+value[0]+','+str(value[1]))
        
def getRetsPerPlayer(atpmatches,qmatches,fmatches, activeplayers, active):
    """returns the retirements per player over his career"""
    allmatcheslist = []
    allmatcheslist.append(atpmatches)
    allmatcheslist.append(qmatches)
    allmatcheslist.append(fmatches)
    allmatches = pd.concat(allmatcheslist)
    
    allmatchesret_group = allmatches[allmatches['score'].str.contains('RET').fillna(False)].groupby('loser_name').size().order(ascending=False)
    allmatcheslost_group = allmatches.groupby('loser_name').size().order(ascending=False)
    allmatcheslost_group = allmatcheslost_group[allmatcheslost_group > 49]
    merged = pd.concat([allmatcheslost_group,allmatchesret_group], axis=1, join = 'inner').reset_index().fillna(0)
    merged.columns = ['name', 'losses' , 'ret']
    merged['percentage'] =  (merged['ret'] * 100 / merged['losses']).round(2)
    merged = merged.sort(['percentage','losses'], ascending=False)
    
    
    if (active):
        activeplayerslist = [row[0] for row in activeplayers]
        merged = merged[(merged['name'].isin(activeplayerslist))]

    print(merged.to_csv(sys.stdout,index=False))   
    
def youngestChallengerWinners(matches):
    """returns youngest challenger match winners"""
    matches = matches[(matches['tourney_level'] == 'C') & (matches['round'] == "R32") & (matches['winner_age'] < 17)]
    matches_grouped = matches.groupby('winner_name', as_index=False).apply(lambda g: g.loc[g.winner_age.idxmin()])
    matches_grouped['winner_age'] = matches_grouped['winner_age'].round(1)
    print(type(matches_grouped['winner_rank'][0]))
    matches_grouped['loser_rank'] = matches_grouped['loser_rank'].fillna(0.0).astype('int')
    matches_grouped['winner_rank'] = matches_grouped['winner_rank'].fillna(0.0).astype('int')
    print(matches_grouped[['tourney_date','tourney_name','winner_name','winner_age', 'winner_rank', 'loser_name', 'loser_rank', 'loser_entry']].sort(['winner_age'], ascending=[True]).to_csv(sys.stdout,index=False, sep='\t'))    
    
def getStreaks(atpmatches):
    """detects streaks in players' careers.
    in the next lines some parameters can be changed."""
    #how many wins allowed during the streak? only tested for 0 and 1.
    #GAPS_ALLOWED=1
    GAPS_ALLOWED=0
    #max rank of player when the streak was started
    #MAX_RANK=20
    MAX_RANK=2000
    #define streak length
    MIN_STREAK_LENGTH = 20
    #winning streak or losing streak?
    #WINS = False
    WINS = True
    
    if (WINS):
        NAME_COLUMN = 'winner_name'
        RANK_COLUMN = 'winner_rank'
    else:
        NAME_COLUMN = 'loser_name'
        RANK_COLUMN = 'loser_rank' 
        
    #change tourney_level in next line!
    #atpmatches = atpmatches[(atpmatches['tourney_date'] >= 19900000) & (atpmatches['tourney_level'] != 'D')]
    atpmatches = atpmatches[(atpmatches['tourney_date'] >= 19900000) & (atpmatches['tourney_level'] == 'S')]
    
    #counting wins and losses (and adding as a column) for each player (for filter later to increase speed of algorithm)
    atpmatches['wfreq'] = atpmatches.groupby('winner_name')['winner_name'].transform('count')
    atpmatches['lfreq'] = atpmatches.groupby('loser_name')['loser_name'].transform('count')

    #only include players with more than 30 wins and 10 losses (to increase algorithm speed by reducing number of players)
    wplayers = atpmatches[atpmatches['wfreq'] > 30]['winner_name'].tolist()
    lplayers = atpmatches[atpmatches['lfreq'] > 10]['loser_name'].tolist()
    players = set(wplayers+lplayers)

    streaks = []
    for player in players:
        playerFullName = player
        matches = atpmatches[((atpmatches['winner_name'] == playerFullName) | (atpmatches['loser_name'] == playerFullName))]
        matches['round'] = pd.Categorical(matches['round'], categories = ["RR", "R128", "R64", "R32", "R16", "QF", "SF", "F", "W"])
        matches = matches.sort(['tourney_date', 'round'])
                
        wins_cnt=0
        streak = 0

        for i in range(len(matches)):
            #get losing streak
            if (matches.iloc[i][NAME_COLUMN] == playerFullName):
                if (streak == 0):
                    startdate = matches.iloc[i]['tourney_date']
                    startrank = matches.iloc[i][RANK_COLUMN]
                streak = streak+1
            else:
                #win
                wins_cnt=wins_cnt+1
                if (wins_cnt==1):
                    win_pos = i+1
                
                if (wins_cnt > GAPS_ALLOWED):  
                    if (streak > MIN_STREAK_LENGTH):
                        streaks.append([playerFullName,startdate,startrank,streak,GAPS_ALLOWED])
                    streak = 0
                    wins_cnt=0
                    i = win_pos+1
            
#old version of the function            
#         for index, match in matches.iterrows():
#             #get losing streak
#             if (match['loser_name'] == playerFullName):
#                 if (streak == 0):
#                     startdate = match['tourney_date']
#                     startrank = match['loser_rank']
#                 streak = streak+1
#                 tempmatch = match
#                 continue
#             else:
#                 #save streak because of win
#                 if (streak > 6):
#                     streaks.append([playerFullName,startdate,startrank,streak])
#                 streak = 0
#                 continue

        #to get ongoing streaks or players who ended career with a losing streak
        if (streak > MIN_STREAK_LENGTH):
            streaks.append([playerFullName,startdate,startrank,streak,GAPS_ALLOWED])
        streak = 0
    #do some styling (for streak-starts where we dont have a ranking (possibly due to WC awarded) we enter 9999 as a streak-ranking-start
    #so in order to include them MAX_RANK needs to be set accordingly
    streaks = [[row[0],row[1],9999 if math.isnan(row[2]) else int(row[2]),row[3],row[4]] for row in streaks]

    #sort by date first because it's secondary sort index
    streaks.sort(key=itemgetter(1),reverse=True)
    #sort by streak length
    streaks.sort(key=itemgetter(3),reverse=True)   
    streaks = [y for y in streaks if int(y[2]) <= MAX_RANK]
    for streak in streaks:
        print(streak[0]+','+str(streak[1])+','+str(streak[2])+','+str(streak[3])+','+str(streak[4]))
        
def get1seedWinners(matches):
    """calculates how often the first seed won an ATP tournament"""
    fmatches = matches[(matches['tourney_level'] == 'A') & (matches['round'] == 'F') & (matches['tourney_date'] >= 19910000)]
    wseed1matches = fmatches[(fmatches['winner_seed'] == 1)]
    print(len(fmatches))
    print(len(wseed1matches))
    
    matches = matches[(matches['tourney_level'] == 'A') & (matches['tourney_date'] >= 19910000)]
    cntfirstseedstten = 0
    firstseedsttenwinner = 0
    cntfirstseedgtten = 0
    firstseedgttenwinner = 0
    cntfirstseedgttwenty = 0
    firstseedgttwentywinner = 0
    
    tourney_group = matches.groupby('tourney_id')
    for tname, tdf in tourney_group:
        #print(tname)
        if (len(tdf[(tdf['winner_seed'] == 1)] > 0)):
            firstseedrank = tdf[(tdf['winner_seed'] == 1)].iloc[0]['winner_rank']
        elif (len(tdf[(tdf['loser_seed'] == 1)] > 0)):
            firstseedrank = tdf[(tdf['loser_seed'] == 1)].iloc[0]['loser_rank']
            
        if not (math.isnan(firstseedrank)):
            if (firstseedrank < 11):
                cntfirstseedstten+=1
                if (len(tdf[(tdf['winner_seed'] == 1) & (tdf['round'] == 'F')] > 0)):
                    firstseedsttenwinner+=1
            if (firstseedrank > 10):
                cntfirstseedgtten+=1
                if (len(tdf[(tdf['winner_seed'] == 1) & (tdf['round'] == 'F')] > 0)):
                    firstseedgttenwinner+=1
            if (firstseedrank > 20):
                cntfirstseedgttwenty +=1
                if (len(tdf[(tdf['winner_seed'] == 1) & (tdf['round'] == 'F')] > 0)):
                    firstseedgttwentywinner+=1

    print('cntfirstseedstten: ' + str(cntfirstseedstten))
    print('firstseedsttenwinner: ' +  str(firstseedsttenwinner))    
    print('cntfirstseedgtten: ' + str(cntfirstseedgtten))
    print('firstseedgttenwinner: ' +  str(firstseedgttenwinner))
    print('cntfirstseedgttwenty: ' +  str(cntfirstseedgttwenty))
    print('firstseedgttwentywinner: ' +  str(firstseedgttwentywinner))
    
def getseedWinners(matches):
    """calculates how often the seeds won an ATP tournament"""
    groundmatches = matches[(matches['tourney_level'] == 'A') & (matches['round'] == 'F') & (matches['tourney_date'] >= 19910000) & ((matches['draw_size'] == 32) | (matches['draw_size'] == 28))]
    wseed1matches = groundmatches[(groundmatches['winner_seed'] == 1)]
    wseed2matches = groundmatches[(groundmatches['winner_seed'] == 2)]
    wseed3matches = groundmatches[(groundmatches['winner_seed'] == 3)]
    wseed4matches = groundmatches[(groundmatches['winner_seed'] == 4)]
    wseed5matches = groundmatches[(groundmatches['winner_seed'] == 5)]
    wseed6matches = groundmatches[(groundmatches['winner_seed'] == 6)]
    wseed7matches = groundmatches[(groundmatches['winner_seed'] == 7)]
    wseed8matches = groundmatches[(groundmatches['winner_seed'] == 8)]
    wunseedseedmatches = groundmatches[(groundmatches['winner_seed'].isnull())]
    print(len(groundmatches))
    print('{} - {}{}'.format('1 seed', np.round(len(wseed1matches)*100/len(groundmatches), decimals=2),'%'))
    print('{} - {}{}'.format('2 seed', np.round(len(wseed2matches)*100/len(groundmatches), decimals=2),'%'))
    print('{} - {}{}'.format('3 seed', np.round(len(wseed3matches)*100/len(groundmatches), decimals=2),'%'))
    print('{} - {}{}'.format('4 seed', np.round(len(wseed4matches)*100/len(groundmatches), decimals=2),'%'))
    print('{} - {}{}'.format('5 seed', np.round(len(wseed5matches)*100/len(groundmatches), decimals=2),'%'))
    print('{} - {}{}'.format('6 seed', np.round(len(wseed6matches)*100/len(groundmatches), decimals=2),'%'))
    print('{} - {}{}'.format('7 seed', np.round(len(wseed7matches)*100/len(groundmatches), decimals=2),'%'))
    print('{} - {}{}'.format('8 seed', np.round(len(wseed8matches)*100/len(groundmatches), decimals=2),'%'))
    print('{} - {}{}'.format('unseeded', np.round(len(wunseedseedmatches)*100/len(groundmatches), decimals=2),'%'))
    print('------')
    
 


def highestRankedAustriansInR16(matches):
    """returns the highest ranked austrians in an R16 of an ATP tournament.
    parameters in the next lines can be changed to make it work for different rounds and countries."""
    smatches = matches[(matches['tourney_level'] == 'A') & (matches['round'] =='R32') & (matches['winner_ioc'] =='AUT') & (matches['winner_rank'] > 300) & ((matches['draw_size'] == 28) | (matches['draw_size'] == 32))]
    bmatches = matches[(matches['tourney_level'] == 'A') & (matches['round'] =='R64') & (matches['winner_ioc'] =='AUT') & (matches['winner_rank'] > 300) & ((matches['draw_size'] == 56) | (matches['draw_size'] == 48) | (matches['draw_size'] == 64))]
    mergelist = [smatches, bmatches]
    matches = pd.concat(mergelist)
    matches = matches.sort(['winner_rank'], ascending=False)
    print(matches[['tourney_name','tourney_date','winner_name', 'winner_rank', 'loser_name', 'loser_rank', 'loser_entry']].to_csv(sys.stdout,index=False))
    
def mostRetsInTourneyPerPlayer(matches):
    """returns tournaments where a player benefitted of a RET or W.O. more than once"""
    matches = matches[(matches['tourney_level'] == 'A') | (matches['tourney_level'] == 'M') | (matches['tourney_level'] == 'G')]
    matches['score'].astype('str')
    matches['count'] = matches.groupby(['tourney_id', 'winner_name'])['score'].transform(lambda x: x[(x.str.contains('RET|W').fillna(False))].count())
    matches = matches[(matches['count'] > 1)]
    print(matches[['tourney_name','tourney_date','winner_name', 'count']].drop_duplicates().to_csv(sys.stdout,index=False))
    
def mostWCs(matches):
    """finds players with most WCs"""
    matches = matches[(matches['tourney_date']  > 20140000)]
    matches = matches[(matches['tourney_level'] == 'A') | (matches['tourney_level'] == 'M')| (matches['tourney_level'] == 'G')]
    matches = matches[(matches['winner_entry'] == 'WC') | (matches['loser_entry'] == 'WC')]
    
    wcw_group = matches.groupby(['tourney_id', 'winner_name']).apply(lambda x: (x['winner_entry'] == 'WC').sum())
    wcl_group = matches.groupby(['tourney_id', 'loser_name']).apply(lambda x: (x['loser_entry'] == 'WC').sum())
    
    scores = pd.DataFrame({'wcw' : wcw_group, 'wcl': wcl_group}).fillna(0)
    scores[['wcw', 'wcl']] = scores[['wcw', 'wcl']].astype(int)
    scores.index.names = ['tourney','pname']
    scores['wcs'] = scores['wcw'] + scores['wcl']
    scores = scores[(scores['wcs'] > 0)]
    scores = scores.groupby('pname')
#     scores[['wins', 'losses', 'rets']] = scores[['wins', 'losses', 'rets']].astype(int)
#     scores.index.names = ['year','pname']
#     scores = scores[(scores['rets'] > 3)]
#     scores['matches'] = scores['wins']+scores['losses']
#     scores['percentage'] = np.round((scores['rets'] * 100 / scores['wins']).astype(np.double), decimals=2)
#     scores = scores.reindex_axis(['matches', 'wins','losses','rets','percentage'], axis=1)
#     scores = scores.sort(['rets'], ascending=False)
    print(scores.to_csv(sys.stdout,index=True))
    
   
def mostRetsPerYear(matches):
    """finds players with most RETs received per year"""
    matches = matches[(matches['tourney_level'] == 'A') | (matches['tourney_level'] == 'M')| (matches['tourney_level'] == 'G')]
    matches['score'].astype('str')
    matches['tourney_date'].apply(str)
    matches['year'] = matches['tourney_date'].astype(str)
    matches['year'] = matches['year'].str[:4]
    
    w_group = matches.groupby(['year', 'winner_name']).size()
    l_group = matches.groupby(['year', 'loser_name']).size()
    #ret_group = matches.groupby(['year', 'winner_name']).apply(lambda x: (x['score'].str.contains('RET|W').fillna(False)).sum())
    ret_group = matches.groupby(['year', 'winner_name']).apply(lambda x: (x['score'].str.contains('RET').fillna(False)).sum())

    scores = pd.DataFrame({'wins' : w_group, 'losses' : l_group, 'rets' : ret_group}).fillna(0)
    scores[['wins', 'losses', 'rets']] = scores[['wins', 'losses', 'rets']].astype(int)
    scores.index.names = ['year','pname']
    scores = scores[(scores['rets'] > 3)]
    scores['matches'] = scores['wins']+scores['losses']
    scores['percentage'] = np.round((scores['rets'] * 100 / scores['wins']).astype(np.double), decimals=2)
    scores = scores.reindex_axis(['matches', 'wins','losses','rets','percentage'], axis=1)
    scores = scores.sort(['rets'], ascending=False)
    print(scores.to_csv(sys.stdout,index=True))
    
    
def oldestWinnerATP(atpmatches,qmatches):
    """returns tournaments with old match winners"""
    mergelist = [atpmatches, qmatches]
    matches = pd.concat(mergelist)
    matches = matches[(matches['tourney_level'] == 'A') | (matches['tourney_level'] == 'Q') | (matches['tourney_level'] == 'M')  | (matches['tourney_level'] == 'G')]
    matches = matches[(matches['winner_age']) > 38]
    #matches = matches[(matches['tourney_date']) > 19903000]
    matches = matches.sort(['winner_age'], ascending=False)
    print(matches[['tourney_name', 'tourney_date', 'round', 'winner_name', 'winner_age', 'loser_name', 'score']].drop_duplicates().to_csv(sys.stdout,index=False))
    
    
def bestNonChampion(players,ranks): 
    """finds highest ranked players without a title"""
    players.columns = ['id', 'fname', 'lname', 'hand', 'dob', 'country']
    players['full_name'] = players['fname'] + ' ' + players['lname']
    ranks.columns = ['ranking_date', 'rank', 'player_id', 'points']
    ranks = ranks[(ranks['rank'] < 41)]
    join = pd.merge(ranks,players, left_on='player_id', right_on='id')
    join["fullname"] = join["fname"] + ' ' + join["lname"]
    join = join[(join['ranking_date'] > datetime.date(1991, 1, 1))]
    
    #join['tournament_wins'] = join.apply(lambda x: len(matches[(matches['tourney_date'] < x['ranking_date']) & (matches['winner_name'] == x['fullname'])]), axis=1)
    join = join.groupby('fullname').apply(get_date_wins) #groupby to increase speed compared to previous line
    join = join[(join['tournament_wins'] == 0)].sort(['rank'], ascending=True)
    print(join[['fullname', 'ranking_date', 'rank', 'tournament_wins']].to_csv(sys.stdout,index=False))
    
def getZeroBreakPointChampions(atpmatches):
    """finds tournament winners who faces zero break points over the course of a tournament"""
    matches = atpmatches[((atpmatches['tourney_level'] == 'A') | (atpmatches['tourney_level'] == 'G') | (atpmatches['tourney_level'] == 'M')) & (atpmatches['tourney_date'] >= 19910000)]
    #matches = matches[(matches['tourney_id'] == '2015-891') | (matches['tourney_id'] == '2015-407')]
    matches['w_broken'] = matches['w_bpFaced'] - matches['w_bpSaved']
    matches = matches.reset_index().groupby('tourney_id').apply(get_winner_name)
    matches = matches[(matches['winner_name'] == matches['twname'])]
    matches['sum_broken'] = matches.groupby('tourney_id')['w_broken'].transform(np.sum)
    matches = matches.sort(['sum_broken','tourney_date'], ascending=[True,False])
    print(matches[['tourney_id', 'tourney_name', 'tourney_level', 'winner_name', 'sum_broken']].drop_duplicates().to_csv(sys.stdout,index=False))

    
def get_winner_name(in_group):
    """helper function"""
    try:
        wname =  in_group[(in_group['round'] == 'F')].iloc[[0]]['winner_name'].values[0]
        in_group['twname'] = wname
    except IndexError:
        in_group['twname'] = 'none'
    return in_group

def easiestOpponents(atpmatches):
    """finds players who had the highest ranked opponents over a complete tournament"""
    matches = atpmatches[(atpmatches['tourney_level'] == 'G')]
    
    matches = matches[(matches['round'] == 'R128') | (matches['round'] == 'R64') | (matches['round'] == 'R32') | (matches['round'] == 'R16')]
    
    #filter players who played against Q or WC (who potentially have high rankings
    #matches = matches[(matches['loser_entry'] != 'WC') & (matches['loser_entry'] != 'Q') & (matches['loser_entry'] != 'LL')]
    
    
    matches = matches.groupby(['tourney_date','winner_name']).filter(lambda x: len(x) > 3)
    matches['ranksum'] = matches.groupby(['tourney_date','winner_name'])['loser_rank'].transform(lambda x: x.sum())
    matches = matches[(matches['ranksum'] > 450)]
    matches = matches.sort(['tourney_date','winner_name'], ascending=True)
    print(matches[['tourney_name','tourney_date','winner_name', 'round', 'loser_name', 'loser_rank', 'loser_entry', 'ranksum']].drop_duplicates().to_csv(sys.stdout,index=False))
    

def wcwinner(qmatches):
    """finds Q winners who were WCs"""
    matches = qmatches[(qmatches['round'] == 'F') & (qmatches['winner_entry'] == 'WC')]
    #filter out seeded WCs
    matches = matches[(matches['winner_seed'].isnull())]
    print(matches[['tourney_name','tourney_date','winner_name', 'winner_entry', 'winner_rank']].to_csv(sys.stdout,index=False))
    
def titlesataage(atpmatches):
    """calculates how many titles a player had at a certain age"""
    matches = atpmatches[(atpmatches['round'] == 'F') & (atpmatches['winner_age'] < 22.5)]
    matches['titles'] = matches.groupby('winner_name')['winner_name'].transform('count')
    matches = matches[(matches['titles'] > 3)]
    print(matches[['winner_name', 'titles']].drop_duplicates().to_csv(sys.stdout,index=False))
    
def get_streaks(x):
    """helper function"""
    #x['streak'] = x.groupby( (x['l_breaks'] != 0).cumsum()).cumcount() + ( (x['l_breaks'] != 0).cumsum() == 0).astype(int)
    x=x.reset_index()
    for i, row in x.iterrows():
        #print(i)
        if i != 0:
            if row['l_breaks'] == 0:
                x.ix[i, 'streak'] = x.ix[i-1, 'streak'] + 1        
        else:
            if row['l_breaks'] == 0:
                x.ix[i, 'streak'] = 1 
    return x

def get_streaks2(df):
    """helper function"""
    df=df.reset_index()
    df['streak2'] = (df['l_breaks'] == 0).cumsum()
    df['cumsum'] = np.nan
    df.loc[df['l_breaks'] != 0, 'cumsum'] = df['streak2']
    df['cumsum'] = df['cumsum'].fillna(method='ffill')
    df['cumsum'] = df['cumsum'].fillna(0)
    df['streak'] = df['streak2'] - df['cumsum']
    df.drop(['streak2', 'cumsum'], axis=1, inplace=True)
    return df
    

def consecutivlosseswithoutbreaks(atpmatches):
    """finds matches where players had consecutive losses without getting broken"""
    #atpmatches = atpmatches[(atpmatches['loser_name'] == 'John Isner')]
    atpmatches = atpmatches.sort('tourney_date')
    atpmatches['l_breaks'] = atpmatches['l_bpFaced']-atpmatches['l_bpSaved']
    atpmatches['streak'] = 0
    atpmatches = atpmatches.groupby('loser_name').apply(get_streaks2)
    atpmatches = atpmatches[(atpmatches['streak'] >1)]
    atpmatches = atpmatches.sort('tourney_date')
    print(atpmatches[['tourney_date', 'tourney_name','winner_name', 'loser_name', 'score', 'l_bpSaved', 'l_bpFaced', 'streak']].to_csv(sys.stdout,index=False))
    
def curse(row):
    """helper function"""
    if row['previous_loser'] == 'Rafael Nadal':
        if row['previous_winner'] == row['winner_name']:
            val = 0
        elif row['previous_winner'] != row['winner_name']:
            val = 1
    else:
        val = -1
        
    return val
    
def findnadals(group):
    """helper function"""
    #print(group.iloc[[0]]['tourney_date'].values[0])
    group = group.sort('rank')
    group['previous_loser'] = group['loser_name'].shift(1)
    group['previous_winner'] = group['winner_name'].shift(1)
    group = group[(group['winner_name'] == 'Rafael Nadal') | (group['loser_name'] == 'Rafael Nadal') | (group['previous_loser'] == 'Rafael Nadal')]
    group.loc[group['previous_loser'] != 'Rafael Nadal', 'previous_loser'] = ''
    group = group[(group['previous_loser'] == 'Rafael Nadal') | ((group['loser_name'] == 'Rafael Nadal') & (group['round'] != 'F'))]
    if (len(group) > 0): 
        group['curse'] = group.apply(curse, axis=1)
        #print(group[['tourney_date', 'tourney_name', 'round', 'winner_name', 'loser_name', 'previous_loser', 'curse', 'score']].to_csv(sys.stdout,index=False))
    #print('in group type: ' + str(type(group)))
    return group

    
def losetonadalafterwin(atpmatches):
    """finds matches of players who lost to nadal after they beat him (nadal curse)"""
    
    round_dict = { "R16": 9,
                   "W": 13,
                   "F": 12,
                   "RR": 8,
                   "R64": 6,
                   "R128": 5,
                   "QF": 10,
                   "SF": 11,
                   "R32": 7
                  };
                  

    
    atpmatches = atpmatches[(atpmatches['tourney_date'] > datetime.date(1995,1,1))]
    w_set = set(atpmatches['winner_name'])
    l_set = set(atpmatches['loser_name'])
    namelist = w_set.union(l_set)  
    namelist = sorted(namelist)
    namelist.remove('Novak Djokovic')
    namelist.remove('Andy Murray')
    namelist.remove('Roger Federer')
    namelist.remove('Rafael Nadal')
    #namelist = namelist.remove('Novak Djokovic'), 'Roger Federer', 'Andy Murray'])
    #namelist = ['Borna Coric', 'Steve Darcis']
    
    atpmatches['rank'] = atpmatches['round'].map(round_dict)
    atpmatches = atpmatches.sort('tourney_date')
    
    #get list of nadal tournaments
    nadal_tourneys = atpmatches[(atpmatches['winner_name'] == 'Rafael Nadal') | (atpmatches['loser_name'] == 'Rafael Nadal')]['tourney_id']
    nadal_tourneys = nadal_tourneys.tolist()
    resultmatches = pd.DataFrame()
    
    for name in namelist:
        print(name)
        matches = atpmatches[((atpmatches['winner_name'] == name) | (atpmatches['loser_name'] == name)) & atpmatches['tourney_id'].isin(nadal_tourneys)]
        matches = matches.groupby(['tourney_id']).apply(findnadals)
        resultmatches = resultmatches.append(matches)
        
    resultmatches['curse'] = resultmatches['curse'].astype(int)
    resultmatches = resultmatches.sort(['tourney_date', 'rank'], ascending=[True,True])
    print(resultmatches[['tourney_date', 'tourney_name', 'round', 'winner_name', 'loser_name', 'previous_loser', 'curse', 'score']].drop_duplicates().to_csv(sys.stdout,index=False))
    print(resultmatches[['tourney_date', 'tourney_name', 'round', 'winner_name', 'loser_name', 'curse', 'score']].drop_duplicates().to_csv(sys.stdout,index=False))
    
def countseeds(group):
    """helper function"""
    group['cnt'] = len(group[(group['loser_seed'] < 6) | (group['winner_seed'] < 6)])
    #print(len(group[(group['loser_seed'] < 6) | (group['winner_seed'] < 6)]))
    #print('--')
    return group
        
    
def fouroffiveseedsgone(atpmatches):
    """finds tournaments where four of five seeds lost until R16"""
    matches = atpmatches[(atpmatches['tourney_level'] == 'M') & (atpmatches['round'] == 'R16')]
    matches = matches.reset_index().groupby(['tourney_id']).apply(countseeds)
    matches = matches[(matches['cnt'] < 2)]
    matches['url'] = 'http://www.protennislive.com/posting/' +matches['tourney_date'].dt.strftime('%Y') + '/' + matches['tourney_id'].str[-3:] + '/mds.pdf'
    print(matches[['tourney_date', 'tourney_name', 'cnt', 'url']].drop_duplicates().to_csv(sys.stdout,index=False))
    
def createOpponentCol(x,name):
    """helper function"""
    if (x['winner_name'] == name):
        return x['loser_name']
    else:
        return x['winner_name']
    
def createOpponent2Col(x,name):
    """helper function"""
    if (x['winner_name'] == name):
        return 1
    else:
        return 0
     
def lossStreaks(group):
    """helper function"""
    group = group.sort('tourney_date')
    group['streak2'] = (group['opponent_loss'] == 0).cumsum()
    group['cumsum'] = np.nan
    group.loc[group['opponent_loss'] == 1, 'cumsum'] = group['streak2']
    group['cumsum'] = group['cumsum'].fillna(method='ffill')
    group['cumsum'] = group['cumsum'].fillna(0)
    group['streak'] = group['streak2'] - group['cumsum']
    group['streak'] = group['streak'].astype('int')
    group = group[group['streak'] != 0]
    #group.drop(['streak2', 'cumsum'], axis=1, inplace=True)    
    group = group.groupby('cumsum').filter(lambda x: len(x) > 1)
    if (len(group) > 1):
        print(group[['tourney_date', 'tourney_name', 'winner_name', 'loser_name', 'score', 'streak']].to_csv(sys.stdout,index=False))
    return group

def backtobacklosses(atpmatches,name):
    """finds back to back losses"""
    matches=atpmatches[(atpmatches['winner_name'] == name) | (atpmatches['loser_name'] == name)]
    matches['opponent_name'] = matches.apply(lambda x: createOpponentCol(x, name), axis=1)
    matches['opponent_loss'] = matches.apply(lambda x: createOpponent2Col(x, name), axis=1)
    #matches = matches[matches['opponent_name'] == 'Novak Djokovic']
    matches = matches.reset_index().groupby('opponent_name').apply(lossStreaks)
    
def titlesdefended(atpmatches):
    """calculates how often titles were successfully defended"""
    matches = atpmatches[(atpmatches['tourney_level'] == 'A')]
    matches['rawid'] = matches['tourney_id'].str[5:]
    matches1415 = matches[((matches['tourney_date'] < 20140000) & (matches['tourney_date'] > 20111200))]
    matches1415['tourney_date'] = matches1415['tourney_date'].astype('str')
    #TODO: check if winner of 2014 is also playing in 2015. if yes: keep tournament, if not, drop it!
    matches1415 = matches1415.groupby(matches1415['rawid'])#.filter(lambda x: len(x['tourney_name']) == 2)
    #print(len(matches1415))
    
    defender_tourneys = matches1415.apply(tryingtodefend)
    print(len(defender_tourneys))
    defender_tourneys = defender_tourneys[defender_tourneys['del'] == 0]
    print(len(defender_tourneys))
    defender_tourneys = defender_tourneys[['rawid','triedtodefend','successfullydefended','madeituntil']]
    print(len(defender_tourneys))
    defender_tourneys = defender_tourneys.drop_duplicates()
    print('{} {}'.format('tournaments:', len(defender_tourneys)))
    print('{} {}'.format('triedtodefend:', len(defender_tourneys[defender_tourneys['triedtodefend'] == 1])))
    print('{} {}'.format('successfullydefended:', len(defender_tourneys[defender_tourneys['successfullydefended'] == 1])))
    print('{} {}'.format('% triedtodefend:', np.round(len(defender_tourneys[defender_tourneys['triedtodefend'] == 1])*100/len(defender_tourneys)),2))
    print('{} {}'.format('% successfullydefended of all tournaments:', np.round(len(defender_tourneys[defender_tourneys['successfullydefended'] == 1])*100/len(defender_tourneys)),2))
    print('{} {}'.format('% successfullydefended of tries:', np.round(len(defender_tourneys[defender_tourneys['successfullydefended'] == 1])*100/len(defender_tourneys[defender_tourneys['triedtodefend'] == 1])),2))
    print('{} {}'.format('until F:', np.round(len(defender_tourneys[defender_tourneys['madeituntil'] == 'F'])*100/len(defender_tourneys[defender_tourneys['triedtodefend'] == 1])),2))
    print('{} {}'.format('until SF:', np.round(len(defender_tourneys[defender_tourneys['madeituntil'] == 'SF'])*100/len(defender_tourneys[defender_tourneys['triedtodefend'] == 1])),2))
    print('{} {}'.format('until QF:', np.round(len(defender_tourneys[defender_tourneys['madeituntil'] == 'QF'])*100/len(defender_tourneys[defender_tourneys['triedtodefend'] == 1])),2))
    print('{} {}'.format('until R16:', np.round(len(defender_tourneys[defender_tourneys['madeituntil'] == 'R16'])*100/len(defender_tourneys[defender_tourneys['triedtodefend'] == 1])),2))
    print('{} {}'.format('until R32:', np.round(len(defender_tourneys[defender_tourneys['madeituntil'] == 'R32'])*100/len(defender_tourneys[defender_tourneys['triedtodefend'] == 1])),2))
    print('{} {}'.format('until R64:', np.round(len(defender_tourneys[defender_tourneys['madeituntil'] == 'R64'])*100/len(defender_tourneys[defender_tourneys['triedtodefend'] == 1])),2))
    

    #defender_tourneys.groupby['rawid']
    #print(defender_tourneys[['rawid','tourney_id','triedtodefend','successfullydefended','madeituntil']].to_csv(sys.stdout,index=False))
    #matches1415 = matches1415.groupby(matches1415['rawid']).apply(defending)


def tryingtodefend(group):
    """helper function"""
    try:
        #print(group.name)
        #print(group.iloc[[0]]['tourney_name'])
        #print(len(group))
        #print('----')
        group = group.sort('tourney_date', ascending=True)
        group_copy = group[(group['round'] == 'F')]
        #get new/old year
        newer_year = str(group_copy.iloc[[len(group_copy)-1]]['tourney_date'].values[0])[:4]
        older_year = str(group_copy.iloc[[0]]['tourney_date'].values[0])[:4]
        oldwinnername = group_copy.iloc[[0]]['winner_name'].values[0]
        tourneyname = group_copy.iloc[[0]]['tourney_name'].values[0]
#         print(oldwinnername)
#         print(older_year)
#         print(newer_year)
#         print(tourneyname)
#         print(len(group_copy))
        
        #del because tournament was only held in one year
        if (len(group_copy) == 1):
            group['del'] = 1
        else:
            group['del'] = 0
            
        #delete big four
        if ((oldwinnername == 'Rafael Nadal') | (oldwinnername == 'Andy Murray') | (oldwinnername == 'Novak Djokovic') | (oldwinnername == 'Roger Federer')):
             group['del'] = 1
            
            

        #does oldwinner play in new year?
        newmatches = group[group['tourney_date'].str.startswith(newer_year) & ((group['loser_name'] == oldwinnername) | (group['winner_name'] == oldwinnername))]
        if (len(newmatches) == 0):
            group['triedtodefend'] = 0
            group['successfullydefended'] = 0
            group['madeituntil'] = 0
            
        else:
            group['triedtodefend'] = 1
            #successfully defended?
            if (len(newmatches[(newmatches['round'] == 'F') & (newmatches['winner_name'] == oldwinnername)]) > 0):
                group['successfullydefended'] = 1
                group['madeituntil'] = 'W'
            else:
                group['successfullydefended'] = 0
                if (len(newmatches[(newmatches['round'] == 'F') & (newmatches['loser_name'] == oldwinnername)]) > 0):
                    group['madeituntil'] = 'F'
                elif (len(newmatches[(newmatches['round'] == 'SF') & (newmatches['loser_name'] == oldwinnername)]) > 0):
                    group['madeituntil'] = 'SF'
                elif (len(newmatches[(newmatches['round'] == 'QF') & (newmatches['loser_name'] == oldwinnername)]) > 0):
                    group['madeituntil'] = 'QF'
                elif (len(newmatches[(newmatches['round'] == 'R16') & (newmatches['loser_name'] == oldwinnername)]) > 0):
                    group['madeituntil'] = 'R16'
                elif (len(newmatches[(newmatches['round'] == 'R32') & (newmatches['loser_name'] == oldwinnername)]) > 0):
                    group['madeituntil'] = 'R32'
                elif (len(newmatches[(newmatches['round'] == 'R64') & (newmatches['loser_name'] == oldwinnername)]) > 0):
                    group['madeituntil'] = 'R64'
        
        
        #print(group[['tourney_id','tourney_name','triedtodefend','successfullydefended','madeituntil']].drop_duplicates().to_csv(sys.stdout,index=False))

        #newwinnername = group_copy.iloc[[1]]['winner_name'].values[0]
        #print(newwinnername)
        
        #get list of all new players

        #print(group[['tourney_date', 'round', 'winner_name', 'prev_winner_winner','prev_winner_runnerup', 'deftitle', 'defrunnerup']].to_csv(sys.stdout,index=False))
        #print(group[['tourney_date', 'round', 'winner_name']].to_csv(sys.stdout,index=False))
        return group
    except IndexError:
        group['include'] = False
        return group

def defending(group):
    """helper function"""
    #print(group.name)
    #print(group.iloc[[0]]['tourney_name'])
    #print(len(group))
    print('----')
    group = group.sort('tourney_date', ascending=True)
    group['prev_winner_winner'] = group['winner_name'].shift()
    group['prev_winner_runnerup'] = group['loser_name'].shift()
    group['deftitle'] = group.apply(f, axis=1)
    group['defrunnerup'] = group.apply(g, axis=1)
    print(group[['rawid', 'winner_name', 'prev_winner_winner','prev_winner_runnerup', 'deftitle', 'defrunnerup']].to_csv(sys.stdout,index=False))
    return group

def f(row):
    """helper function"""
    if row['prev_winner_winner'] == row['winner_name']:
        val = 1
    else:
        val = 0
    return val

def g(row):
    """helper function"""
    if row['prev_winner_runnerup'] == row['winner_name']:
        val = 1
    else:
        val = 0
    return val

def titlessurface(atpmatches):
    """calculates titles per surface"""
    matches = atpmatches[atpmatches['round'] == 'F']
    matches['year'] = matches.tourney_id.str[:4]
    matches['tourney_date'] =matches['tourney_date'].astype('str') 
    matches['month'] = matches.tourney_date.str[4:6]
    matches['month'] = matches['month'].astype('int')
    matches = matches[matches['month'] < 7]
    matches = matches.reset_index().groupby(['year','winner_name'])
    surface_winners = matches.apply(calcsurfaces)

def calcsurfaces(group):
    """helper function"""
    #print(group.iloc[[0]]['year'].values[0])
    #print(group.iloc[[0]]['winner_name'].values[0])
    #print(len(group[group['surface'] == "Clay"]))
    #print(len(group[group['surface'] == "Grass"]))
    #print(len(group[group['surface'] == "Hard"]))
    
    if ((len(group[group['surface'] == "Clay"])) > 0) & ((len(group[group['surface'] == "Hard"])) > 0) & ((len(group[group['surface'] == "Grass"])) > 0):
        print(group[['year','winner_name', 'surface','tourney_name', 'tourney_date']].to_csv(sys.stdout,index=False))
        

def matchesPerLastNameAndRound(matches):
    """finds matches of brothers"""
    matches = matches[(matches['round']=='F') & (matches['tourney_date'] > 19900000)]
    matches['winner_lastname'] = matches['winner_name'].str.split(' ').str[-1] + matches['winner_ioc']
    matches['loser_lastname'] = matches['loser_name'].str.split(' ').str[-1] + matches['loser_ioc']
    matches = matches.reset_index().groupby('tourney_id')
    result = matches.apply(playernames)
    result = result[(result['winner_brother'] == 1) | (result['loser_brother'] == 1)]
    result2 = []
    for index, row in result.iterrows():
        output1=''
        output2=''
        
        if (row['winner_brother'] == 1) & (row['loser_brother'] == 1):
            output1+=row['winner_name']
            output2+=row['loser_name']
            result2.append([row['tourney_name'], row['tourney_date'], output1,1])
            result2.append([row['tourney_name'], row['tourney_date'], output2,1])
        else:
            if (row['winner_brother'] == 1):
                output1+=row['winner_name']
            if (row['loser_brother'] == 1):
                output1+=row['loser_name']
            
            result2.append([row['tourney_name'], row['tourney_date'], output1,0])

    resultdf = pd.DataFrame(result2)
    resultdf[4] = resultdf[2].shift(1)
    resultdf = resultdf[1:]
    resultdf = resultdf.iloc[::2]
    resultdf = resultdf[[0,1,2,4,3]]
    #print matches.to_string(columns=['tourney_name','tourney_date','winner_name', 'loser_name'])
    print(resultdf.to_csv(sys.stdout,index=False))

def playernames(group):
    """helper function"""
    group['unique_sum'] = len(group['winner_lastname'].unique()) + len(group['loser_lastname'].unique())
    temp = [group['winner_lastname'], group['loser_lastname']]
    mergelist = pd.concat(temp)
    mergelist = mergelist[(mergelist != 'LopezESP') & (mergelist != 'JohanssonSWE') & (mergelist != 'CostaESP')  & (mergelist != 'FerreiraRSA') ]
    dups = mergelist.duplicated(keep=False)
    dups2 = mergelist[mergelist.duplicated(keep=False)]
    dups3 = dups2.str[0:-3]
    
    group['winner_lastname'] = group['winner_lastname'].str[0:-3]
    group['loser_lastname'] = group['loser_lastname'].str[0:-3]
    
    group['winner_brother'] = np.where(group['winner_lastname'].isin(dups3), 1, 0)
    group['loser_brother'] = np.where(group['loser_lastname'].isin(dups3), 1, 0)

    return group

def bestNeverQFWin(matches, rankings,activeplayers):
    """finds players who never won a QF (+ streaks)"""
    #matches = matches[matches['tourney_date'] >  datetime.date(2012,12,29)]
    qfmatches = matches[(matches['round']=='QF')]
    qfmatches = qfmatches.sort_values(by='tourney_date')
    qfgmatches = qfmatches.groupby('winner_name').first().reset_index()
    
    #print(len(matches))
    #print(matches[[ 'round', 'tourney_date','tourney_name']].head(1000))
    
    #reused code start (would be better to make a function out of this) 
    global joinedrankingsdf
    #join rankings with playernames
    dirname = ".."
    playersDB = dirname + "/atp_players.csv"
    
    rankings.columns = ['date', 'rank', 'id','points']
    rankings = rankings[rankings['rank'] < 100]
    playersdf = pd.read_csv(playersDB,index_col=None,header=None,encoding = "ISO-8859-1")
    playersdf.columns = ['id', 'fname', 'lname','hand','dob','country']
    playersdf["fullname"] = playersdf["fname"] + ' ' + playersdf["lname"]
    playersdf = playersdf.sort_values(by='fullname')
    
    qfstreaks = pd.DataFrame()
    
    #filter players to increase speed of following iteration
    r16matches = matches[(matches['round']=='R16')]
    r16players = set(r16matches['winner_name'])
    filtered_players = playersdf[playersdf['fullname'].isin(r16players)]
    print(len(playersdf))
    print(len(filtered_players))
    #calc initial losing streaks
    for index, player in filtered_players.iterrows():
        name = player['fullname']
        print(name)
        
        qfmatchesplayer = qfmatches[(qfmatches['winner_name'] == name) | (qfmatches['loser_name'] == name)]
        qfmatchesplayer['player'] = name
        #print(len(qfmatchesplayer))
        if (len(qfmatchesplayer)) > 0:
            streak = 1
            for index, row in qfmatchesplayer.iterrows():
                if (row['winner_name'] == row['player']):
                    streak=0
                    qfmatchesplayer.set_value(index,'streak',streak)
                elif (row['loser_name'] == row['player']) & (streak==1):
                    qfmatchesplayer.set_value(index,'streak',streak)
                else:
                    streak=0
                    qfmatchesplayer.set_value(index,'streak',streak)
            qfmatchesplayer=qfmatchesplayer[qfmatchesplayer['streak']==1]
            qfstreaks = qfstreaks.append(qfmatchesplayer)
            #print(qfmatchesplayer[['player','winner_name', 'loser_name','streak']].to_csv(sys.stdout,index=False))
    counts_df = pd.DataFrame(qfstreaks.groupby('player').size().sort_values().rename('counts'))
    
    print(counts_df)
            
    joinedrankingsdf = pd.merge(rankings,playersdf, on='id')
    joinedrankingsdf["fullname"] = joinedrankingsdf["fname"] + ' ' + joinedrankingsdf["lname"]
    #reused code end
       
    res = []
    #get all rankings for each player row
    for index, match in qfgmatches.iterrows():
        name = match['winner_name']
        date = match['tourney_date']
        print(name)
        try:
            counts = counts_df[counts_df.index == name]['counts'].values[0]
        except:
            counts = 0
        playerres = joinedrankingsdf[(joinedrankingsdf['fullname'] == name) & (joinedrankingsdf['date'] <= date)]
        try:
            minrank = playerres.loc[playerres['rank'].idxmin()]
            #print(minrank[['fullname','date', 'rank']])
 
            res.append([minrank['fullname'], minrank['rank'], minrank['date'].strftime('%Y.%m.%d'),1,counts])
        except:
            continue
        
        
    #add players who still did not win a QF
    qfmatches = matches[(matches['round']=='QF')]
    qfwinners = set(qfmatches['winner_name'])
    qfmatches = qfmatches.groupby('loser_name').filter(lambda g: (len(g[(~g['loser_name'].isin(qfwinners))]) > 0))
    counts_df = pd.DataFrame(qfmatches.groupby('loser_name').size().sort_values().rename('counts'))
 
    for index, match in counts_df.iterrows():
        name = match.name
        counts = match['counts']
        playerres = joinedrankingsdf[(joinedrankingsdf['fullname'] == name)]
        try:
            minrank = playerres.loc[playerres['rank'].idxmin()]
            res.append([minrank['fullname'], minrank['rank'], minrank['date'].strftime('%Y.%m.%d'),0,counts])
        except:
            continue
          
    res = sorted(res, key=itemgetter(1))
    pprint.pprint(res)
    
    activeplayerslist = [row[0] for row in activeplayers]
     
    for r in res:
        if (r[0] in activeplayerslist):
            print("{: >20} {: >20} {: >20} {: >20} {: >20}".format(*r))
        
    
    
def listAllTimeNoQFWins(matches):
    """lists players who never won a QF"""
    qfmatches = matches[(matches['round']=='QF')]
    qfwinners = set(qfmatches['winner_name'])
    qfmatches = qfmatches.groupby('loser_name').filter(lambda g: (len(g[(~g['loser_name'].isin(qfwinners))]) > 0))
    counts = qfmatches.groupby('loser_name').size().sort_values()
    print(counts)
    
def setstats(atpmatches):
    """for a player calculates specific set statistics"""
    name='Gael Monfils'
    matches=atpmatches[(atpmatches['winner_name'] == name) | (atpmatches['loser_name'] == name)]
    matches=matches[matches['tourney_date'] >  datetime.date(2014,12,28)]
    #setfilter
    matches = matches[(matches['score'].str.count('-') == 3) | (matches['score'].str.count('-') == 2)]
    #norets
    matches = matches[~matches['score'].str.contains('RET|W').fillna(False)]
    matches['sets_analysis'] = matches.apply(analyzeSets, axis=1)
    matches['sets_won'], matches['sets_lost'], matches['first'], matches['res']  = zip(*matches['sets_analysis'].map(lambda x: x.split(',')))
    matches['sets_won'] = matches['sets_won'].astype('int')
    matches['sets_lost'] = matches['sets_lost'].astype('int')
    matches['first'] = matches['first'].astype('int')
    print('sets won: ' + str(matches['sets_won'].sum()))
    print('sets lost: ' + str(matches['sets_lost'].sum()))
    print('first sets won: ' + str(matches['first'].sum()))
    print('cb analysis:\n' + str(matches['res'].value_counts(sort=False)))
    print('# of matches: ' + str(len(matches)))
    #print(matches[['score','sets_won', 'sets_lost', 'first','winner_name', 'loser_name']].to_csv(sys.stdout,index=False))

    
    
def analyzeSets(row):
    """helper function"""
    sets = row['score'].split(' ')
    won=0
    lost=0
    first=0
    res=0
    print(sets)
    for idx,set in enumerate(sets):
        setscore = set.split('-')
        if (len(setscore)>1):
            #clean tb scores
            if( '(' in setscore[0]):
                setscore[0]=setscore[0][0]
            if( '(' in setscore[1]):
                setscore[1]=setscore[1][0]
            if(row['winner_name'] == 'Gael Monfils'):
                print('player winner')
                if((int(setscore[0])>int(setscore[1])) & (int(setscore[0]) > 5)):
                    won=won+1
                    if(idx==0):
                        first=1
                elif(int(setscore[0])<int(setscore[1])):
                    lost=lost+1
            else:
                print('player loser')
                if((int(setscore[0])<int(setscore[1])) & (int(setscore[1]) > 5)):
                    won=won+1
                    if(idx==0):
                        first=1
                elif((int(setscore[0])>int(setscore[1]))):
                    lost=lost+1 
        print(setscore)
    #ersten gewonnen und gewonnen = 0
    if ((first==1) & (won>lost)):
        res=0
    #ersten verloren und gewonnen = 1
    if ((first==0) & (won>lost)):
        res=1
    #ersten gewonnen und verloren = 2
    if ((first==1) & (won<lost)):
        res=2
    #ersten verloren und verloren = 3
    if ((first==0) & (won<lost)):
        res=3
    print(str(won)+','+str(lost)+','+str(first)+','+str(res))
    return(str(won)+','+str(lost)+','+str(first)+','+str(res))
            
    
        
        
def geth2hforplayerswrapper(atpmatches,qmatches):
    """helper function"""
    #geth2hforplayer(atpmatches,"Roger Federer")
    atpmatches = atpmatches.append(qmatches)
    names = atpmatches[atpmatches['winner_rank'] < 100]
    names = names.winner_name.unique()
    for name in names:
        geth2hforplayer(atpmatches,name)
        
def getwnonh2hs(atpmatches,qmatches,rankings):
    """calculates head to heads"""
    #todo: could be extended to older players and also show career-overlap (e.g. were 10y together on tour)s
    #make full matches df
    atpmatches = atpmatches.append(qmatches)
    
    global joinedrankingsdf
    #join rankings with playernames
    dirname = ".."
    playersDB = dirname + "/atp_players.csv"
    
    rankings.columns = ['date', 'rank', 'id','points']
    playersdf = pd.read_csv(playersDB,index_col=None,header=None,encoding = 'iso-8859-1')
    playersdf.columns = ['id', 'fname', 'lname','hand','dob','country']
    
    joinedrankingsdf = pd.merge(rankings,playersdf, on='id')
    joinedrankingsdf["fullname"] = joinedrankingsdf["fname"] + ' ' + joinedrankingsdf["lname"]
    
    #get newest top-n rankings
    joinedrankingsdf = joinedrankingsdf[(joinedrankingsdf.date == joinedrankingsdf.date.max()) & (joinedrankingsdf['rank'] < 51)]
    
    #for each player in the rankings calculate the h2hs and then compare players in the h2h list with top-n players...
    playernames = joinedrankingsdf['fullname']
    playernameslist = playernames.tolist()
    
    #dfs for calculating match-number of players
    wins = atpmatches.groupby('winner_name').count()
    losses = atpmatches.groupby('loser_name').count()
    
    for player in playernames:
        h2hlist = geth2hforplayer(atpmatches,player)
        h2hnames = [row[0] for row in h2hlist]
        noh2hs = [x for x in playernameslist if x not in h2hnames]
        
        player1WL = losses[losses.index == player]['score'].values[0]+wins[wins.index == player]['score'].values[0]

        #show rank for player in iteration and show string
        for noh2h in noh2hs:
            player2WL = losses[losses.index == noh2h]['score'].values[0]+wins[wins.index == noh2h]['score'].values[0]
            if (noh2h != player):
                print(player + ';' + str(joinedrankingsdf[(joinedrankingsdf['fullname'] == player)]['rank'].values[0]) + ';' + str(player1WL) + ';' + noh2h + ';' + str(joinedrankingsdf[(joinedrankingsdf['fullname'] == noh2h)]['rank'].values[0]) + ';' + str(player2WL))

def getTop100ChallengerPlayersPerWeek(qmatches):
    """finds top 100 challenger players per week"""
    matches = qmatches[(qmatches['tourney_level'] == 'C') & (qmatches['round'] == 'R32') & (qmatches['tourney_date'] > datetime.date(2000,1,1))]
    matches['top100'] = matches.apply(top100, axis=1)
    matches['count'] = matches.groupby(['tourney_date'])['top100'].transform(lambda x: x.sum())
    matches['tcount'] = matches.groupby(['tourney_date'])['tourney_name'].transform(lambda x: x.nunique())
    matches = matches.sort(['count'], ascending=False)
    print(matches[['tourney_date', 'count','tcount']].drop_duplicates().to_csv(sys.stdout,index=False))
    #print(matches['top100'].head())


def top100(row):
    """helper function"""
    if ((row['winner_rank'] < 101) & (row['loser_rank'] < 101)):
     val = 2
    elif ((row['loser_rank'] < 101) | (row['winner_rank'] < 101)):
     val = 1
    else:
     val = 0
    #print(val)
    return val

def showTourneysOfDate(qmatches,year,month,day):
    """for a date shows the tournaments which were played at this date"""
    matches = qmatches[(qmatches['tourney_level'] == 'S') & (qmatches['round'] == 'R32') & (qmatches['tourney_date'] == datetime.date(year,month,day))]
    matches = matches[(matches['loser_rank'] < 151) | (matches['winner_rank'] < 151)]
    print(matches[['tourney_id', 'winner_name', 'winner_rank','loser_name','loser_rank']].drop_duplicates().to_csv(sys.stdout,index=False))
    print(matches[['tourney_date', 'tourney_name','tourney_id']].drop_duplicates().to_csv(sys.stdout,index=False))
    
def titles(matches):
    """calculates titles per player"""
    matches = matches[(matches['round'] == 'F')]
    matches['titles'] = matches.groupby('winner_name')['winner_name'].transform('count')
    matches = matches[(matches['titles'] > 15)]
    matches = matches.sort(['titles'], ascending=False)
    print(matches[['winner_name', 'titles']].drop_duplicates().to_csv(sys.stdout,index=False))
    

def lowestRankedTitlists(matches):
    """finds the lowest ranked titlists"""
    matches = matches[(matches['tourney_level'] == 'C') & (matches['round'] == 'F') & (matches['winner_rank'] > 600)]
    matches = matches.sort(['tourney_date'], ascending=False)
    matches['winner_rank'] = matches['winner_rank'].astype('int')
    matches['winner_age'] = matches['winner_age'].round(2)
    print(matches[['tourney_date', 'tourney_name', 'winner_name', 'winner_rank', 'winner_age']].to_csv(sys.stdout,index=False))
    
def gamesconcededpertitle(matches):
    """calculates how many games a player lost per title"""
    matches=matches[matches['tourney_date'] > datetime.date(2000,1,1)]
    matches = matches[(matches['tourney_level'] == 'S')]
    matches['wcnt'] = matches.groupby(['tourney_id','winner_name'])['winner_name'].transform('count')
    matches = matches[matches['wcnt'] == 5]
    matches['sets_analysis'] = matches.apply(analyzeSetsFutures, axis=1)
    matches['games_won'], matches['games_lost'], matches['rets'] = zip(*matches['sets_analysis'].map(lambda x: x.split(',')))
    
    #convert columns to int for summing up
    matches['games_won'] = matches['games_won'].astype('int')
    matches['games_lost'] = matches['games_lost'].astype('int')
    matches['rets'] = matches['rets'].astype('int')
    
    #calculate the sum over each matches games
    matches['games_won_t'] = matches.groupby(['tourney_id'])['games_won'].transform('sum')
    matches['games_lost_t'] = matches.groupby(['tourney_id'])['games_lost'].transform('sum')
    matches['rets_t'] = matches.groupby(['tourney_id'])['rets'].transform('sum')
    
    #convert columns to int for summing up
    matches['games_won_t'] = matches['games_won_t'].astype('int')
    matches['games_lost_t'] = matches['games_lost_t'].astype('int')
    matches['rets_t'] = matches['rets_t'].astype('int')
    
    
    matches = matches.sort(['games_lost_t'], ascending=True)
    print(matches[['tourney_id', 'winner_name', 'wcnt','games_won_t','games_lost_t','rets_t']].drop_duplicates().to_csv(sys.stdout,index=False))
    
    
def analyzeSetsFutures(row):
    """helper function"""
    #6-4 6-7(5) 6-4
    #setscore[0] sind die vom sieger, setscore[1] sind die vom verlierer
    try:
        sets = row['score'].split(' ')
        gameswonwinner=0
        gameslostwinner=0
        retcount=0
        if 'RET' in row['score']: retcount=1
        if 'W' in row['score']: retcount=1
        #print(sets)
        for idx,set in enumerate(sets):
            setscore = set.split('-')
            if (len(setscore)>1):
                #clean tb scores
                if( '(' in setscore[0]):
                    setscore[0]=setscore[0][0]
                if( '(' in setscore[1]):
                    setscore[1]=setscore[1][0]
                gameswonwinner=gameswonwinner+int(setscore[0])
                gameslostwinner=gameslostwinner+int(setscore[1])
    
        #print(str(gameswonwinner)+','+str(gameslostwinner)+','+str(retcount))
        return(str(gameswonwinner)+','+str(gameslostwinner)+','+str(retcount))
    except:
        return(str(0)+','+str(0)+','+str(0))

def lastTimeGrandSlamCountry(atpmatches):
    """grand slam results per country"""
    matches=atpmatches[(atpmatches['tourney_level'] == 'G') & ((atpmatches['winner_ioc'] == 'NOR') | (atpmatches['loser_ioc'] == 'NOR'))]
    matches = matches.sort(['tourney_date'], ascending=True)
    print(matches[['tourney_date','tourney_name', 'round', 'winner_name', 'loser_name']].to_csv(sys.stdout,index=False))
    
def countunder21grandslam(atpmatches):
    """calculates how many players under 21 were in a grand slam main draw"""
    matches=atpmatches[(atpmatches['tourney_level'] == 'G') & (atpmatches['round'] == 'R128')]
    matches['w_under_21'] = matches.groupby(['tourney_id'])['winner_age'].transform(lambda x: x[x < 21].count())
    matches['l_under_21'] = matches.groupby(['tourney_id'])['loser_age'].transform(lambda x: x[x < 21].count())
    matches = matches.reset_index().groupby(['tourney_id']).apply(concat)

    matches['players_under_21'] = matches['w_under_21']+matches['l_under_21']
    matches['players_under_21'] = matches['players_under_21'].astype('int')
    
    matches['player_names'] = matches['w_under_21_names'] + ',' + matches['l_under_21_names'] 
    #matches = matches.sort(['players_under_21'], ascending=True)
    print(matches[['tourney_id','tourney_name', 'player_names','players_under_21']].drop_duplicates().to_csv(sys.stdout,index=False))
    #print(matches[['tourney_date','tourney_name','players_under_21']].drop_duplicates().to_csv(sys.stdout,index=False))
    

def concat(group):
    """helper function"""
    group['l_under_21_names'] = "%s" % ', '.join(group['loser_name'][group['loser_age'] < 21])
    group['w_under_21_names'] = "%s" % ', '.join(group['winner_name'][group['winner_age'] < 21])
    return group

def countryTitle(atpmatches):
    """calculates titles per country"""
    matches=atpmatches[(atpmatches['round'] == 'F') & ((atpmatches['winner_ioc'] == 'LUX') | (atpmatches['loser_ioc'] == 'LUX'))]
    print(matches[['tourney_date','tourney_name','winner_name','loser_name']].to_csv(sys.stdout,index=False))

def youngGsmatchwinners(atpmatches):
    """calculates young grand slam match winners"""
    matches=atpmatches[(atpmatches['tourney_level'] == 'G') & (atpmatches['winner_age'] < 18)]
    print(matches[['tourney_date','tourney_name','winner_name', 'winner_age', 'loser_name','loser_age']].to_csv(sys.stdout,index=False))
    
def mostPlayersInTop100OfCountry(rankings):
    """calculates how many players of a country are in the top1000"""
    global joinedrankingsdf
    #join rankings with playernames
    #note: working with the rankings: make sure iso8859-1 is set to encoding when parsing and that file is without BOM
    dirname = ".."
    playersDB = dirname + "/atp_players.csv"
    
    rankings.columns = ['date', 'rank', 'id','points']
    playersdf = pd.read_csv(playersDB,index_col=None,header=None,encoding = "ISO-8859-1")
    playersdf.columns = ['id', 'fname', 'lname','hand','dob','country']

    joinedrankingsdf = pd.merge(rankings,playersdf, on='id')
    joinedrankingsdf=joinedrankingsdf[(joinedrankingsdf['date'] > datetime.date(2005,1,1)) & (joinedrankingsdf['rank'] < 101)]
    joinedrankingsdf["fullname"] = joinedrankingsdf["fname"] + ' ' + joinedrankingsdf["lname"]
    joinedrankingsdf['namerank'] = joinedrankingsdf['fullname']+ "," + joinedrankingsdf['rank'].map(str) 
    #joinedrankingsdf["namerank"] = str(joinedrankingsdf["fullname"]) + ',' + str(joinedrankingsdf["rank"])

    joinedrankingsdf['auts'] = joinedrankingsdf.groupby(['date'])['country'].transform(lambda x: x[(x.str.contains('AUT').fillna(False))].count())
    joinedrankingsdf=joinedrankingsdf[(joinedrankingsdf['auts'] > 3) &  (joinedrankingsdf['country'] == 'AUT')]
    joinedrankingsdf = joinedrankingsdf.reset_index().groupby(['date']).apply(concatranknames)

    joinedrankingsdf = joinedrankingsdf.sort(['date'], ascending=True)
    print(joinedrankingsdf[['date', 'country','autnames']].to_csv(sys.stdout,index=False))
    
def concatranknames(group):
    """helper function"""
    group['autnames'] = "%s" % ', '.join(group['namerank'][group['country'] == 'AUT'])
    return group

def topSeedsGS(atpmatches):
    """calculates performance of top seeds at grand slams"""
    matches=atpmatches[(atpmatches['tourney_level'] == 'G')]
    resmatches = matches.reset_index().groupby(['tourney_id']).apply(calcSeeds)
    resmatches = resmatches[resmatches['topseeds'] == 0]
    print(resmatches[['tourney_date', 'tourney_name','topseeds']].drop_duplicates().to_csv(sys.stdout,index=False))
    res2matches = resmatches[((resmatches['round'] == 'R16') | (resmatches['round'] == 'R32') | (resmatches['round'] == 'R128') | (resmatches['round'] == 'R64')) & (resmatches['loser_seed'] < 3) & (resmatches['topseeds'] == 0)]
    print(res2matches[['tourney_date', 'tourney_name','round','winner_name','loser_name','loser_seed','loser_rank','score']].to_csv(sys.stdout,index=False))


def calcSeeds(group):
    """helper function"""
    group['topseeds'] = len(group[(group['round'] == 'QF') & ((group['winner_seed'] < 3) | (group['loser_seed'] < 3))])
    return group   

def top10winstitlist(atpmatches):
    """calculates how many top 10 wins a titlist had in the tournament he won"""
    #matches = atpmatches[(atpmatches['tourney_date'] > 20000101) & (atpmatches['tourney_level'] != 'D') & (atpmatches['round'] != 'RR') & (atpmatches['tourney_id'] != '2008-438')]
    matches = atpmatches[(atpmatches['tourney_date'] > 19900101) & (atpmatches['tourney_level'] == 'A') & (atpmatches['round'] != 'RR') & (atpmatches['tourney_id'] != '2008-438')]
    matches = matches.reset_index().groupby(['tourney_id']).apply(calcTop10WinsForTitlist)
    matches = matches[(matches['titlistrank'] > 10) & (matches['titlistname'] == matches['winner_name']) & (matches['titlisttop10wins'] > 2)]
    print(matches[['tourney_date', 'tourney_name','tourney_level','titlisttop10wins','round','winner_name','winner_rank','loser_name','loser_rank','score']].to_csv(sys.stdout,index=False))

    
def calcTop10WinsForTitlist(group):
    """helper function"""
    #print(group['tourney_id'])
    titlistname = group[(group['round'] == 'F')].iloc[[0]]['winner_name'].values[0]
    titlistrank = group[(group['round'] == 'F')].iloc[[0]]['winner_rank'].values[0]
    group['titlistname'] = titlistname
    group['titlistrank'] = titlistrank
    group['titlisttop10wins'] = len(group[(group['winner_name'] == titlistname) & (group['loser_rank'] < 11)])
    return group   

def findLLwhoWOdinQ(atpmatches,qmatches):
    """find if LL wo'd in FQR"""
    resultlist = list()
    tourney_group = atpmatches.groupby('tourney_id')
    for tname, tdf in tourney_group:
        found1=False
        found2=False
        #first_case finds where a LL won against a Q in a main draw (MD)
        first_case = tdf[(tdf['winner_entry'] == 'LL')]
        #iterating over first_case matches
        for index, match in first_case.iterrows():
            first_case_results = qmatches[(qmatches['tourney_name'] == match['tourney_name']+ ' Q') & ((qmatches['round'] =='Q2') | (qmatches['round'] =='Q3')) & (match['winner_name'] == qmatches['loser_name'])]
            if (len(first_case_results.index) > 0):
                #if results were found, add the MD match to the result list
                resultlist.append(first_case_results)
    
          
        #second_case finds where a LL lost against a Q in a main draw (MD)  
        second_case = tdf[(tdf['loser_entry'] == 'LL')]
        for index, match in second_case.iterrows():
            second_case_results = qmatches[(qmatches['tourney_name'] == match['tourney_name']+ ' Q') & ((qmatches['round'] =='Q2') | (qmatches['round'] =='Q3')) & (match['loser_name'] == qmatches['loser_name'])]
            if (len(second_case_results.index) > 0):
                #if results were found, add the MD match to the result list
                resultlist.append(second_case_results)
    
           
    result = pd.concat(resultlist).sort(['tourney_date'], ascending=False)
    print(result[['tourney_name','tourney_date','round','winner_name','winner_entry', 'loser_name','loser_entry','score']].to_csv(sys.stdout,index=False))
    
def highestRanked500finalist(atpmatches):
    """finds highest ranked ATP 500 finalists"""
    matches = atpmatches[(atpmatches['tourney_date'] > datetime.date(2008,12,20))]
    
    
    #atp 500
    #if draw size = 32, then 8 seeds
    #if draw size = 48, then 16 seeds
    #if draw size = 56, then 16 seeds
    tourney500names = ['Rotterdam', 'Rio de Janeiro', 'Acapulco', 'Dubai', 'Barcelona', 'Hamburg', 'Washington', 'Beijing', 'Tokyo', 'Valencia', 'Basel', 'Memphis']
    
    matches500 = matches[matches['tourney_name'].isin(tourney500names)]
    #remove 2014-402 (= memphis) because in 2014 it was a 250
    matches500 = matches500[(matches500['tourney_id'] != '2014-402')]
    matches500 = matches500[(matches500['round'] == 'F')]
    matches500w = matches500[['tourney_date', 'tourney_name','round','winner_name','winner_rank']]
    matches500w['result'] = 'W'
    matches500w.columns = ['tourney_date', 'tourney_name','round','player_name','player_rank','result']
    matches500l = matches500[['tourney_date', 'tourney_name','round','loser_name','loser_rank']]
    matches500l['result'] = 'L'
    matches500l.columns = ['tourney_date', 'tourney_name','round','player_name','player_rank','result']
    final_dfs = [matches500w, matches500l]
    final = pd.concat(final_dfs).sort(['player_rank'], ascending=False)
    final['player_rank'] = final['player_rank'].astype(int)
    print(final[['tourney_date', 'tourney_name','player_name','player_rank','result']].to_csv(sys.stdout,index=False,sep= '-'))


def ageBetweenPlayers(atpmatches,qmatches,fmatches):
    """finds age between players"""
    LIMIT = 40
    allmatcheslist = []
    allmatcheslist.append(atpmatches)
    allmatcheslist.append(qmatches)
    allmatcheslist.append(fmatches)
    allmatches = pd.concat(allmatcheslist)   
    allmatches['agediff'] = allmatches['winner_age'] - allmatches['loser_age']
    allmatches = allmatches[(allmatches['agediff'] < LIMIT*-1) | (allmatches['agediff'] > LIMIT)]
    allmatches['agediff'] = allmatches['agediff'].apply(lambda x: x*-1 if x < 0 else x) 
    allmatches['winner_age'] = allmatches['winner_age'].round(1)
    allmatches['loser_age'] = allmatches['loser_age'].round(1)
    allmatches['agediff'] = allmatches['agediff'].round(1)
    print(allmatches[['tourney_id', 'tourney_name','winner_name', 'winner_age', 'loser_name', 'loser_age' , 'agediff']].to_csv(sys.stdout,index=False))


def percentageOfSeedWinnersinQ(qmatches):
    """finds percentage of seeded winners in Q"""
    #i only want atp 250 qualies here, so i need to filter the 4 grand slams
    #i dont have to filter 500 and 1000 qualies because later they dont have a Q3
    matches = qmatches[(qmatches['tourney_level'] == 'Q') & (qmatches['round'] == 'Q3') & (qmatches['tourney_name'] != 'US Open Q') & (qmatches['tourney_name'] != 'Wimbledon Q') & (qmatches['tourney_name'] != 'Roland Garros Q') & (qmatches['tourney_name'] != 'Australian Open Q')]
    matches['seedw'] = matches.groupby('tourney_id')['winner_seed'].transform(lambda x: x[(x > 0)].count())
    matches = matches[['tourney_id', 'tourney_name','seedw']].drop_duplicates()
    counts = matches['seedw'].value_counts()
    dfcnt = pd.DataFrame(counts, columns=['cnt'])
    dfcnt['sum'] = dfcnt['cnt'].sum()
    dfcnt['percentage'] = dfcnt['cnt']*100/dfcnt['sum'].round(1)
    #print(matches[['tourney_id', 'tourney_name','seedw']].to_csv(sys.stdout,index=False))
    print(dfcnt)

def getRankedDict(dict):
    """helper function"""
    rank, count, previous, result = 0, 0, None, {}
    for key, num in dict:
        count += 1
        if num != previous:
            rank += count
            previous = num
            count = 0
        result[key] = rank
    return result

def percentagOfQWinners(qmatches):
    """finds the percentage of Q winners"""
    mydict = collections.defaultdict(dict)
    #i only want atp 250 qualies here, so i need to filter the 4 grand slams
    #i dont have to filter 500 and 1000 qualies because later they dont have a Q3
    matches = qmatches[(qmatches['tourney_level'] == 'Q') & (qmatches['round'] == 'Q3') & (qmatches['tourney_name'] != 'US Open Q') & (qmatches['tourney_name'] != 'Wimbledon Q') & (qmatches['tourney_name'] != 'Roland Garros Q') & (qmatches['tourney_name'] != 'Australian Open Q')]
    
    #so matches right now only contains atp250er qualies (because of Q3 filter)
    #i now want all these tourney_ids
    tourneyids = matches['tourney_id'].unique()
    
    matches_group = qmatches[qmatches['tourney_id'].isin(tourneyids)].groupby('tourney_id')
    for tname, tdf in matches_group:
        for index, match in tdf.iterrows():
            mydict[match.tourney_id][match.winner_name] =  9999.0 if math.isnan(match.winner_rank) else match.winner_rank
            mydict[match.tourney_id][match.loser_name] = 9999.0 if math.isnan(match.loser_rank) else match.loser_rank

    for key, value in mydict.items():
        s_data = sorted(value.items(), key=lambda item: item[1])
        result = getRankedDict(s_data)
        rankdict[key] = result
    
    for key,value in rankdict.items():
        for key1, value1 in value.items():
            if (value1 < 9):
                value[key1] = 1
            if ((value1 < 17) & (value1 > 8)):
                value[key1] = 2
            if (value1 > 16):
                value[key1] = 3
                
    matches['group'] = matches.apply(getGroup, axis=1)
    #matches = matches[matches['tourney_id'] == '2015-867']
    tournamentcnt = len(matches['tourney_id'].unique())
    print("250q tournaments: " + str(tournamentcnt))
    matches['groupc'] = matches.groupby('tourney_id')['group'].transform(lambda x: x[(x == 3)].count())
    #print(matches[['tourney_id', 'groupc', 'groupb','groupu']].to_csv(sys.stdout,index=False))
    groupcmatches = matches[matches['groupc'] > 0 ]
    groupctournamentcnt = len(groupcmatches['tourney_id'].unique())
    print("250q tournaments with at least one groupc q: " + str(groupctournamentcnt))
    print("percentage of tournaments with at least one groupc q: " + str(groupctournamentcnt*100/tournamentcnt))

    cntdf = pd.DataFrame([['groupa', len(matches[matches['group'] == 1]),len(matches)],['groupb', len(matches[matches['group'] == 2]),len(matches)],['groupc', len(matches[matches['group'] == 3]),len(matches)]])
    cntdf.columns = ['groupname', 'quantity','sum']
    cntdf['percentage'] = (cntdf['quantity']*100/cntdf['sum']).round(1)
    print(cntdf)
    
    ############
    print('now for full ATP qs:')
    #same as above but only take full q draws
    fullqmatches = qmatches[(qmatches['tourney_level'] == 'Q') & (qmatches['round'] == 'Q1') & (qmatches['tourney_name'] != 'US Open Q') & (qmatches['tourney_name'] != 'Wimbledon Q') & (qmatches['tourney_name'] != 'Roland Garros Q') & (qmatches['tourney_name'] != 'Australian Open Q')]
    fullqmatches['winners'] = fullqmatches.groupby('tourney_id')['winner_name'].transform(lambda x: x[(x.str.contains('').fillna(False))].count())
    fullqmatches = fullqmatches[fullqmatches['winners'] == 16]
    fullqmatcheslist = fullqmatches['tourney_id'].unique()
    fullqmatchesfinals = qmatches[(qmatches['tourney_id'].isin(fullqmatcheslist)) & (qmatches['tourney_level'] == 'Q') & (qmatches['round'] == 'Q3') & (qmatches['tourney_name'] != 'US Open Q') & (qmatches['tourney_name'] != 'Wimbledon Q') & (qmatches['tourney_name'] != 'Roland Garros Q') & (qmatches['tourney_name'] != 'Australian Open Q')]
    tournamentcnt = len(fullqmatchesfinals['tourney_id'].unique())
    print("250q tournaments with full q: " + str(tournamentcnt))
    fullqmatchesfinals['group'] = fullqmatchesfinals.apply(getGroup, axis=1)
    fullqmatchesfinals['groupc'] = fullqmatchesfinals.groupby('tourney_id')['group'].transform(lambda x: x[(x == 3)].count())
    groupcmatches = fullqmatchesfinals[fullqmatchesfinals['groupc'] > 0 ]
    groupctournamentcnt = len(groupcmatches['tourney_id'].unique())
    print("250q tournaments with at least one groupc q: " + str(groupctournamentcnt))
    print("percentage of tournaments with at least one groupc q: " + str(groupctournamentcnt*100/tournamentcnt))
    cntdf = pd.DataFrame([['groupa', len(fullqmatchesfinals[fullqmatchesfinals['group'] == 1]),len(fullqmatchesfinals)],['groupb', len(fullqmatchesfinals[fullqmatchesfinals['group'] == 2]),len(fullqmatchesfinals)],['groupc', len(fullqmatchesfinals[fullqmatchesfinals['group'] == 3]),len(fullqmatchesfinals)]])
    cntdf.columns = ['groupname', 'quantity','sum']
    cntdf['percentage'] = (cntdf['quantity']*100/cntdf['sum']).round(1)
    print(cntdf)
    ################
    
def getGroup(row):
    """helper function"""
    tid = row['tourney_id']
    name = row['winner_name']
    group = rankdict[tid][name]
    return group

def findSmallestQDraws(qmatches):
    """finds the smallest Q draws"""
    matches = qmatches[(qmatches['tourney_level'] == 'Q') & (qmatches['round'] == 'Q3') & (qmatches['tourney_name'] != 'US Open Q') & (qmatches['tourney_name'] != 'Wimbledon Q') & (qmatches['tourney_name'] != 'Roland Garros Q') & (qmatches['tourney_name'] != 'Australian Open Q')]
    tourneyids = matches['tourney_id'].unique()
    
    matches = qmatches[(qmatches['tourney_id'].isin(tourneyids))]
    matches = matches.reset_index().groupby('tourney_id').apply(myfunc)
    matches = matches.sort('player_sums', ascending=True)
    print(matches[['tourney_id', 'tourney_name','player_sums']].drop_duplicates().to_csv(sys.stdout,index=False))

def myfunc(group):
    """helper function"""
    #get all players into a set
    w_set = set(group['winner_name'])
    l_set = set(group['loser_name'])
    #u_set contains all names of participating players
    group['player_sums'] = len(w_set.union(l_set))
    return group



def youngestCombinedAge(atpmatches,qmatches,fmatches):
    """finds youngest combined age"""
    LIMIT = 40
    allmatcheslist = []
    allmatcheslist.append(atpmatches)
    allmatcheslist.append(qmatches)
    allmatcheslist.append(fmatches)
    allmatches = pd.concat(allmatcheslist)
    
    allmatches['agecombined'] = allmatches['winner_age'] + allmatches['loser_age']
    allmatches = allmatches[(allmatches['agecombined'] < 37)]
    allmatches['winner_age'] = allmatches['winner_age'].round(1)
    allmatches['loser_age'] = allmatches['loser_age'].round(1)
    allmatches['agecombined'] = allmatches['agecombined'].round(1)
    allmatches = allmatches.sort('agecombined', ascending=True)
    print(allmatches[['tourney_id', 'tourney_name','winner_name', 'winner_age', 'loser_name', 'loser_age' , 'agecombined']].to_csv(sys.stdout,index=False))
    

#this needs to be global for percentagOfQWinners() to work
rankdict = collections.defaultdict(dict)


joinedrankingsdf = pd.DataFrame()
#reading ATP level matches. The argument defines the path to the match files.
#since the match files are in the parent directory we provide ".." as an argument
#atpmatches = readATPMatches("..")
atpmatches = readATPMatchesParseTime("..")

#reading Challenger + ATP Q matches
#qmatches = readChall_QATPMatches("..")
#qmatches = readChall_QATPMatchesParseTime("..")
#fmatches = readFMatches("..")
#fmatches = readFMatchesParseTime("..")
#rankings = readAllRankings("..")

#the following lines make use of methods defined above this file. just remove the hash to uncomment the line and use the method.
#matchesPerCountryAndRound(matches)
#findLLQmultipleMatchesAtSameTournament(atpmatches,qmatches)
#bestLLinGrandSlams(atpmatches)
#numberOfSetsLongerThan(atpmatches,2,130)
#geth2hforplayerswrapper(atpmatches,qmatches)
#getwnonh2hs(atpmatches,qmatches,rankings)
#getTop100ChallengerPlayersPerWeek(qmatches)
#getTop100ChallengerPlayersPerWeek(fmatches)
#showTourneysOfDate(fmatches,2011,10,3)
#geth2hforplayer(atpmatches,"Roger Federer")
#getStreaks(fmatches)
#activeplayers = getActivePlayers("..")
#getWinLossByPlayer(fmatches,activeplayers,False)
#seedRanking(atpmatches)
#qualifierSeeded(fmatches)
#rankofQhigherthanlastSeed(atpmatches)
#highRankedQLosers(qmatches,atpmatches)
#avglastseedrank(atpmatches)
#getBestQGrandSlamPlayer(qmatches,rankings)
#getShortestFiveSetter(atpmatches)
#getworstlda(atpmatches)
#getCountriesPerTournament(qmatches)
#getRetsPerPlayer(atpmatches,qmatches,fmatches,activeplayers,False)
#youngestChallengerWinners(qmatches)
#bestNonChampion(players,ranks)
#fedR4WimbiTime(atpmatches)
#youngFutures(fmatches)
#rankingPointsOfYoungsters(players,ranks)
#highestRankedAustriansInR16(atpmatches)
#mostRetsInTourneyPerPlayer(atpmatches)
#mostRetsPerYear(atpmatches)
#mostWCs(atpmatches)
#oldestWinnerATP(atpmatches,qmatches)
#getAces(qmatches)
#getRets(fmatches)
#get1seedWinners(atpmatches)
#getseedWinners(atpmatches)
#getZeroBreakPointChampions(atpmatches)
#easiestOpponents(atpmatches)
#wcwinner(atpmatches)
#titlesataage(atpmatches)
#consecutivlosseswithoutbreaks(atpmatches)
#losetonadalafterwin(atpmatches)
#fouroffiveseedsgone(atpmatches)
#backtobacklosses(atpmatches,'Rafael Nadal')
#titlesdefended(atpmatches)
#titlessurface(atpmatches)
#matchesPerLastNameAndRound(atpmatches)
#bestNeverQFWin(atpmatches,rankings,activeplayers)
#listAllTimeNoQFWins(atpmatches)
#setstats(atpmatches)
#titles(fmatches)
#lowestRankedTitlists(qmatches)
#gamesconcededpertitle(fmatches)
#lastTimeGrandSlamCountry(atpmatches)
#countunder21grandslam(atpmatches)
#countryTitle(fmatches)
#youngGsmatchwinners(atpmatches)
#mostPlayersInTop100OfCountry(rankings)
#topSeedsGS(atpmatches)
#top10winstitlist(atpmatches)
#findLLwhoWOdinQ(atpmatches,qmatches)
#ageBetweenPlayers(atpmatches,qmatches,fmatches)
#percentageOfSeedWinnersinQ(qmatches)
#percentagOfQWinners(qmatches)
#findSmallestQDraws(qmatches)
#youngestCombinedAge(atpmatches,fmatches,qmatches)
highestRanked500finalist(atpmatches)