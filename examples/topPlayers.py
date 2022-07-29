import csv
from itertools import islice
from collections import OrderedDict

# itertools.ifilter() was removed in Python 3,
# the built-in filter() function provides the same functionality now.
try:
    # Python 2
    from future_builtins import filter
except ImportError:
    # Python 3
    pass

MAINDIR = "../"
with  open(MAINDIR + "atp_players.csv") as pf, open(MAINDIR + "atp_rankings_current.csv") as rf:
    players = OrderedDict((row[0], row) for row in csv.reader(pf))
    rankings = csv.reader(rf)
    for i in islice(rankings, None, 10):
        # now constant work getting row as opposed to 0(n)    
        print(players.get(i[2]))
