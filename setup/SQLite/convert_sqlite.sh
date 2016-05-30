#!/bin/bash

if [ $# -eq 0 ]; then 
   MYDATABASE="atpdatabase.db"
else
   MYDATABASE=$1
fi

echo "Creating $MYDATABASE"

sqlite3 $MYDATABASE << !
CREATE TABLE player (id INT PRIMARY KEY, firstName, lastName, hand, birth, country);
.headers off
.mode csv
.import atp_players.csv player
!
echo "Players Imported"

for i in `ls atp_match*`
do
   sqlite3 $MYDATABASE << !
.headers on
.mode csv
.import $i matches
!
done
echo "Matches Imported"

echo "create table ranking (date,pos INT,player_id INT,pts INT);" | sqlite3 $MYDATABASE

for i in `ls atp_rank*`
do
   sqlite3 $MYDATABASE << !
.headers off
.mode csv
.import $i ranking
!
done

echo "Rankings Imported"

echo "Creating Index"
sqlite3 $MYDATABASE << !
create index rankPlayer ON ranking (player_id);
create index rankPos ON ranking (pos);
create index rankDate ON ranking (date);
create index playerCountry ON player (country);
!
