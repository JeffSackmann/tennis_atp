# Top 100 Average Age

It seems the average of the Top100 ATP players is going up.

## Reading Multiple files
```
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
```

## Reading Players

```
def readPlayers(dirname):
    return pd.read_csv(dirname+"/atp_players.csv",
                       index_col=None,
                       header=None,
                       parse_dates=[4],
                       date_parser=lambda t:parse(t))

```

![Age average top 100 players](https://raw.githubusercontent.com/ppaulojr/tennis_atp/master/examples/Pandas/AverageTop100Age/GraphTop100.png) 
