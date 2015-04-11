#!/usr/bin/env python
import csv
MAINDIR = "../"
pf = open (MAINDIR+"atp_players.csv")
players = [p for p in csv.reader(pf)]
rf = open (MAINDIR+"atp_rankings_current.csv")
rankings = [r for r in csv.reader(rf)]
for i in rankings[:10]:
	player = filter(lambda x: x[0]==i[2],players)[0]
	print "%s(%s),(%s) Points: %s"%(player[2],player[5],player[3],i[3])

