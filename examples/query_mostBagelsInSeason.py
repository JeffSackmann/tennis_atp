import csv

## scans results files to identify players with
## most bagels (6-0 sets won) in a single season

## yrend is inclusive
mw, yrstart, yrend = 'm', 1991, 2015

if mw == 'm':   prefix = 'atp'
else:   prefix = 'wta'

## load files for chosen years
matches = [row for row in csv.reader(open(prefix+'_matches_'+str(yrstart)+'.csv'))]
for yr in range(yrstart+1, yrend+1):
    matches += [row for row in csv.reader(open(prefix+'_matches_'+str(yr)+'.csv'))]

## initial filtering of relevant matches
matches = filter(lambda x: '6-0' in x[27] or '0-6' in x[27], matches)

bagel_years = {}
for m in matches:
    tnyid, tnyname, surf, field, tlev, tdate, mno, wid, wseed, watt, wname, whand, wht, wcc, wage, wrank, wpts = m[:17]
    lid, lseed, latt, lname, lhand, lht, lcc, lage, lrank, lpts, score, bestof, rd = m[17:30]

    if '6-0' in score:
        ## key is yr+player
        wkey = tnyid[:4] + ' ' + wname
        if wkey not in bagel_years:   bagel_years[wkey] = []
        ## for each bagel, add list item with date (mmdd), tourney name, and round
        bagel_years[wkey] += [tdate[4:]+' '+tnyname+' '+rd]*score.count('6-0')

    if '0-6' in score:
        lkey = tnyid[:4] + ' ' + lname
        if lkey not in bagel_years:   bagel_years[lkey] = []
        bagel_years[lkey] += [tdate[4:]+' '+tnyname+' '+rd]*score.count('0-6')        

rows = []
for bc in bagel_years:
    ## show only player-seasons with 10+ bagels
    if len(bagel_years[bc]) >= 10:
        ## find and include metadata for 10th (chronological) bagel
        bagels = sorted(bagel_years[bc])
        tenth_bagel = bagels[9]
        rows.append([bc[:4], bc[5:], len(bagel_years[bc]), tenth_bagel])

## sort by most bagels
rows = sorted(rows, key=lambda x: int(x[2]), reverse=True)

results  = open(prefix+'_bagels_by_year.csv', 'wb')
writer = csv.writer(results)
for row in rows:    writer.writerow(row)
results.close()


