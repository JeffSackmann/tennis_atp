#!/bin/bash

if [ $# -eq 0 ]; then 
   MYDATABASE="atpdatabase"
else
   MYDATABASE=$1
fi

echo "Creating $MYDATABASE"

# Lesgo
createdb $MYDATABASE

# Create players table
psql -c "CREATE TABLE players (
    id serial,
    player_id integer,
    firstname character varying(50),
    lastname character varying(50),
    hand character varying,
    birth character varying(8),
    country character varying(3),
    birth_date date
);" $MYDATABASE

# Populate using players CSV
psql -c "\copy players(player_id, firstname, lastname, hand, birth, country) from '../../atp_players.csv' delimiter ',' csv;" $MYDATABASE
# Set birth_date from birth string
psql -c "UPDATE players SET birth_date=TO_DATE(birth,'YYYYMMDD');" $MYDATABASE

echo "Players Imported"

# Create matches table
psql -c "CREATE TABLE matches (
	id serial,
    tourney_id character varying(20),
    tourney_name character varying(50),
    surface character varying(10),
    draw_size integer,
    tourney_level character varying,
    tourney_date character varying(8),
    match_num integer,
    winner_id integer,
    winner_seed integer,
    winner_entry character varying(10),
    winner_name character varying(50),
    winner_hand character varying,
    winner_ht character varying(10),
    winner_ioc character varying(10),
    winner_age double precision,
    winner_rank integer,
    winner_rank_points integer,
    loser_id integer,
    loser_seed integer,
    loser_entry character varying(10),
    loser_name character varying(50),
    loser_hand character varying,
    loser_ht character varying(10),
    loser_ioc character varying(10),
    loser_age double precision,
    loser_rank integer,
    loser_rank_points integer,
    score character varying(50),
    best_of integer,
    round character varying(10),
    minutes integer,
    w_ace integer,
    w_df integer,
    w_svpt integer,
    w_1stin integer,
    w_1stwon integer,
    w_2ndwon integer,
    w_svgms integer,
    w_bpsaved integer,
    w_bpfaced integer,
    l_ace integer,
    l_df integer,
    l_svpt integer,
    l_1stin integer,
    l_1stwon integer,
    l_2ndwon integer,
    l_svgms integer,
    l_bpsaved integer,
    l_bpfaced integer,
    match_date date
);" $MYDATABASE

# Populate using matches CSVs
for x in $(ls ../../atp_matches_*.csv);
	do psql -c "\COPY matches(tourney_id, tourney_name, surface, draw_size, tourney_level, tourney_date, match_num, winner_id, winner_seed, winner_entry, winner_name, winner_hand, winner_ht, winner_ioc, winner_age, winner_rank, winner_rank_points, loser_id, loser_seed, loser_entry, loser_name, loser_hand, loser_ht, loser_ioc, loser_age, loser_rank, loser_rank_points, score, best_of, round, minutes, w_ace, w_df, w_svpt, w_1stin, w_1stwon, w_2ndwon, w_svgms, w_bpsaved, w_bpfaced, l_ace, l_df, l_svpt, l_1stin, l_1stwon, l_2ndwon, l_svgms, l_bpsaved, l_bpfaced) FROM '$x' DELIMITER ',' CSV HEADER;" $MYDATABASE
done;
# Set match_date using tourney_date string
psql -c "UPDATE matches SET match_date=TO_DATE(tourney_date,'YYYYMMDD');" $MYDATABASE

echo "Matches Imported"

# Create rankings table
psql -c "CREATE TABLE rankings (
	id serial,
    date character varying(8),
    pos integer,
    player_id integer,
    pts integer,
    ranking_date date
);" $MYDATABASE

# Populate using rankings CSVs
for x in $(ls ../../atp_rankings_*s.csv);
	do psql -c "\COPY rankings(date, pos, player_id, pts) FROM '$x' DELIMITER ',' CSV;" $MYDATABASE
done;

# Set match_date using tourney_date string
psql -c "UPDATE rankings SET ranking_date=TO_DATE(date,'YYYYMMDD');" $MYDATABASE

echo "Rankings Imported"