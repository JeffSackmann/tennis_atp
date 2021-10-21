## ATP Tennis Rankings, Results, and Stats

This contains my master ATP player file, historical rankings, results, and match stats.

The player file columns are player_id, first_name, last_name, hand, birth_date, country_code, height (cm).

The columns for the ranking files are ranking_date, ranking, player_id, ranking_points (where available).

ATP rankings are mostly complete from 1985 to the present. 1982 is missing, and rankings from 1973-1984 are only intermittent.

Results and stats: There are up to three files per season: One for tour-level main draw matches (e.g. 'atp_matches_2014.csv'), one for tour-level qualifying and challenger main-draw matches, and one for futures matches.

Most of the columns in the results files are self-explanatory. I've also included a matches_data_dictionary.txt file to spell things out a bit more.

To make the results files easier for more people to use, I've included a fair bit of redundancy with the biographical and ranking files: each row contains several columns of biographical information, along with ranking and ranking points, for both players. Ranking data, as well as age, are as of tourney_date, which is almost always the Monday at or near the beginning of the event.

MatchStats are included where I have them. In general, that means 1991-present for tour-level matches, 2008-present for challengers, and 2011-present for tour-level qualifying. The MatchStats columns should be self-explanatory, but they might not be what you're used to seeing; it's all integer totals (e.g. 1st serves in, not 1st serve percentage), from which traditional percentages can be calculated.

There are some tour-level matches with missing stats. Some are missing because ATP doesn't have them. Others I've deleted because they didn't pass some sanity check (loser won 60% of points, or match time was under 20 minutes, etc). Also, Davis Cup matches are included in the tour-level files, but there are no stats for Davis Cup matches until the last few seasons.

# Doubles

I've added tour-level doubles back to 2000. Filenames follow the convention atp_matches_doubles_yyyy.csv. I may eventually be able to add tour-level doubles from before 2000, as well as lower-level doubles for some years. Most of the columns are the same, though in a different order.

Doubles updates are temporarily suspended as of late 2020.

# Contributing

If you find a bug, please file an issue, and be as specific as possible.

Feel free to correct bugs or fill in missing data via pull requests, but be aware that I will not merge PRs. But if that's the most convenient way for you to submit improvements to the data, that's fine; I can work with that.

If you'd like to contribute to the project, I post "help wanted" [issues](https://github.com/JeffSackmann/tennis_atp/issues), starting with a plea to fill in biographical data such as date of birth.

Also, I encourage everyone to pitch in to the [Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject) by charting pro matches. It's not a direct contribution to this repo, but it is a great way to improve the existing state of tennis data.

# Attention

Please read, understand, and abide by the license below. It seems like a reasonable thing to ask, given the hundreds of hours I've put into amassing and maintaining this dataset. Unfortunately, a few bad apples have violated the license, and when people do that, it makes me considerably less motivated to continue updating.

Also, if you're using this for academic/research purposes (great!), take a minute and cite it properly. It's not that hard, it helps others find a useful resource, and let's face it, you should be doing it anyway.

# License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" href="http://purl.org/dc/dcmitype/Dataset" property="dct:title" rel="dct:type">Tennis databases, files, and algorithms</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="http://www.tennisabstract.com/" property="cc:attributionName" rel="cc:attributionURL">Jeff Sackmann / Tennis Abstract</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.<br />Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/JeffSackmann" rel="dct:source">https://github.com/JeffSackmann</a>.

In other words: Attribution is required. Non-commercial use only. 
