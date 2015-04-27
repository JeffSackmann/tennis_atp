#!/usr/bin/env python
import pandas as pd
import glob
import matplotlib.pyplot as plt
import datetime

pd.options.display.mpl_style = 'default'

def parse(t):
    string_ = str(t)
    try:
        return datetime.date(int(string_[:4]), int(string[4:6]), int(string[6:]))
    except:
        return datetime.date(1900,1,1)
    
def readAllFiles(dirname):
    allFiles = glob.glob(dirname + "/atp_rankings_" + "*.csv")
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

def readPlayers(dirname):
    return pd.read_csv(dirname+"/atp_players.csv",
                       index_col=None,
                       header=None,
                       parse_dates=[4],
                       date_parser=lambda t:parse(t))


ranks = readAllFiles(".")
ranks = ranks[(ranks[1]<100)]
players = readPlayers (".")
plRanks = ranks.merge(players,right_on=0,left_on=2)
plRanks["B"] = plRanks["0_x"] - plRanks[4]
plRanks["B"] = plRanks["B"].astype(int) / (365*24*3600*1000000000.0)
agg = plRanks[["0_x","B"]].groupby("0_x")

agg.mean().to_csv("top100ages.csv")
