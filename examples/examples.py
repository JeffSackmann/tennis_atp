'''
    File name: examples.py
    Description: all sorts of tennis-stats examples based on Jeff's tennis data (see: https://github.com/JeffSackmann/tennis_atp)
    Comment: the approach to do the calculations may not always be the most pythonic way. please forgive :)
    Author: Beta2k
    Python Version: 3
'''
import csv
import datetime
import glob
import sys
import operator
import itertools
import collections
from operator import itemgetter
from collections import OrderedDict
import json
import pandas as pd
import numpy as np
import math
from pandas.core.categorical import Categorical



#util functions
def parse(t):
    ret = []
    for ts in t:
        string_ = str(ts)
        try:
            tsdt = datetime.date(int(string_[:4]), int(string_[4:6]), int(string_[6:]))
        except:
            tsdt = datetime.date(1900,1,1)
        ret.append(tsdt)
    return ret

def readATPMatches(dirname):
    """Reads ATP matches but does not parse time into datetime object"""
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

def readATPMatchesParseTime(dirname):
    """Reads ATP matches and parses time into datetime object"""
    allFiles = glob.glob(dirname + "/atp_matches_" + "????.csv")
    matches = pd.DataFrame()
    list_ = list()
    for filen in allFiles:
        df = pd.read_csv(filen,
                         index_col=None,
                         header=0,
                         parse_dates=[5],
                         date_parser=lambda t:parse(t))
        list_.append(df)
    matches = pd.concat(list_)
    return matches

def readFMatches(dirname):
    """Reads ITF future matches but does not parse time into datetime object"""
    allFiles = glob.glob(dirname + "/atp_matches_futures_" + "????.csv")
    matches = pd.DataFrame()
    list_ = list()
    for filen in allFiles:
        df = pd.read_csv(filen,
                         index_col=None,
                         header=0)
        list_.append(df)
    matches = pd.concat(list_)
    return matches

def readChall_QATPMatchesParseTime(dirname):
    """reads Challenger level + ATP Q matches and parses time into datetime objects"""
    allFiles = glob.glob(dirname + "/atp_matches_qual_chall_" + "????.csv")
    matches = pd.DataFrame()
    list_ = list()
    for filen in allFiles:
        df = pd.read_csv(filen,
                         index_col=None,
                         header=0,
                         parse_dates=[5],
                         date_parser=lambda t:parse(t))
        list_.append(df)
    matches = pd.concat(list_)
    return matches

def readChall_QATPMatches(dirname):
    """reads Challenger level + ATP Q matches but does not parse time into datetime objects"""
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

def readAllRankings(dirname):
    """reads all ranking files"""
    allFiles = glob.glob(dirname + "/atp_rankings_" + "*.csv")
    #allFiles = ['..\\atp_rankings_00s.csv', '..\\atp_rankings_10s.csv']
    ranks = pd.DataFrame()
    list_ = list()
    for filen in allFiles:
        df = pd.read_csv(filen,
                         index_col=None,
                         header=None,
                         parse_dates=[0],
                         date_parser=lambda t:parse(t))
        list_.append(df)
    ranks = pd.concat(list_)
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
    print(sorted(h2hlist, key=itemgetter(1,2)))
    
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
    print matches[['tourney_name','tourney_date','winner_name', 'winner_age', 'loser_name']].to_csv(sys.stdout,index=False)
    
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
    
    orderedRes = (OrderedDict(sorted(results.iteritems(), key=lambda x: x[1]['date'])))
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
    print join[['ranking_date','dob', 'points','rank','readAge',  'full_name', 'country']].to_csv(sys.stdout,index=False)

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
    GAPS_ALLOWED=1
    #max rank of player when the streak was started
    MAX_RANK=20
    #define streak length
    MIN_STREAK_LENGTH = 10
    #winning streak or losing streak?
    WINS = False
    
    if (WINS):
        NAME_COLUMN = 'winner_name'
        RANK_COLUMN = 'winner_rank'
    else:
        NAME_COLUMN = 'loser_name'
        RANK_COLUMN = 'loser_rank' 
        
    atpmatches = atpmatches[(atpmatches['tourney_date'] >= 19900000) & (atpmatches['tourney_level'] != 'D')]
    wplayers = atpmatches['winner_name'].tolist()
    lplayers = atpmatches['loser_name'].tolist()
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
    
def highestRankedAustriansInR16(matches):
    """returns the highest ranked austrians in an R16 of an ATP tournament.
    parameters in the next lines can be changed to make it work for different rounds and countries."""
    smatches = matches[(matches['tourney_level'] == 'A') & (matches['round'] =='R32') & (matches['winner_ioc'] =='AUT') & (matches['winner_rank'] > 300) & ((matches['draw_size'] == 28) | (matches['draw_size'] == 32))]
    bmatches = matches[(matches['tourney_level'] == 'A') & (matches['round'] =='R64') & (matches['winner_ioc'] =='AUT') & (matches['winner_rank'] > 300) & ((matches['draw_size'] == 56) | (matches['draw_size'] == 48) | (matches['draw_size'] == 64))]
    mergelist = [smatches, bmatches]
    matches = pd.concat(mergelist)
    matches = matches.sort(['winner_rank'], ascending=False)
    print matches[['tourney_name','tourney_date','winner_name', 'winner_rank', 'loser_name', 'loser_rank', 'loser_entry']].to_csv(sys.stdout,index=False)
    
def mostRetsInTourneyPerPlayer(matches):
    """returns tournaments where a player benefitted of a RET or W.O. more than once"""
    matches = matches[(matches['tourney_level'] == 'A') | (matches['tourney_level'] == 'M')]
    matches['score'].astype('str')
    matches['count'] = matches.groupby(['tourney_id', 'winner_name'])['score'].transform(lambda x: x[(x.str.contains('RET|W').fillna(False))].count())
    matches = matches[(matches['count'] > 1)]
    print matches[['tourney_name','tourney_date','winner_name', 'count']].drop_duplicates().to_csv(sys.stdout,index=False)
    
def mostRetsPerYear(matches):
    """returns players who benefitted from more than 3 RETs per year"""
    matches = matches[(matches['tourney_level'] == 'A') | (matches['tourney_level'] == 'M')| (matches['tourney_level'] == 'G')]
    matches['score'].astype('str')
    matches['tourney_date'].apply(str)
    matches['year'] = matches['tourney_date'].astype(str)
    matches['year'] = matches['year'].str[:4]
    #matches['count'] = matches.groupby(['year', 'winner_name'])['score'].transform(lambda x: x[(x.str.contains('RET|W').fillna(False))].count())
    matches['ret_count'] = matches.groupby(['year', 'winner_name'])['score'].transform(lambda x: x[(x.str.contains('RET').fillna(False))].count())
    matches['match_count'] = matches.groupby(['year', 'winner_name'])['score'].transform(lambda x: x.count())
    matches = matches[(matches['ret_count'] > 3)]
    matches = matches[['year','winner_name', 'match_count', 'ret_count']].drop_duplicates()
    matches['percentage'] = np.round((matches['ret_count'] * 100 / matches['match_count']).astype(np.double), decimals=2)
    print matches[['year','winner_name', 'match_count', 'ret_count', 'percentage']].drop_duplicates().to_csv(sys.stdout,index=False)
    
def oldestWinnerATP(atpmatches,qmatches):
    """returns tournaments with old match winners"""
    mergelist = [atpmatches, qmatches]
    matches = pd.concat(mergelist)
    matches = matches[(matches['tourney_level'] == 'A') | (matches['tourney_level'] == 'Q') | (matches['tourney_level'] == 'M')  | (matches['tourney_level'] == 'G')]
    matches = matches[(matches['winner_age']) > 38]
    #matches = matches[(matches['tourney_date']) > 19903000]
    matches = matches.sort(['winner_age'], ascending=False)
    print matches[['tourney_name', 'tourney_date', 'round', 'winner_name', 'winner_age', 'loser_name', 'score']].drop_duplicates().to_csv(sys.stdout,index=False)
    
    
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
    print join[['fullname', 'ranking_date', 'rank', 'tournament_wins']].to_csv(sys.stdout,index=False)

    
#joinedrankingsdf = pd.DataFrame()
#reading ATP level matches. The argument defines the path to the match files.
#since the match files are in the parent directory we provide ".." as an argument
atpmatches = readATPMatches("..")
#atpmatches = readATPMatchesParseTime("..")

#reading Challenger + ATP Q matches
#qmatches = readChall_QATPMatches("..")
#qmatches = readChall_QATPMatchesParseTime("..")
#fmatches = readFMatches("..")
#rankings = readAllRankings("..")

#the following lines make use of methods defined above this file. just remove the hash to uncomment the line and use the method.
#matchesPerCountryAndRound(matches)
#findLLQmultipleMatchesAtSameTournament(atpmatches,qmatches)
#bestLLinGrandSlams(atpmatches)
#numberOfSetsLongerThan(atpmatches,2,130)
#geth2hforplayer(atpmatches,"Roger Federer")
#getStreaks(atpmatches)
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
#oldestWinnerATP(atpmatches,qmatches)
#getAces(qmatches)
#getRets(fmatches)
get1seedWinners(atpmatches)