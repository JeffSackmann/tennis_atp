## Scripts for SQLite usage

### Usage

```bash
./convert_sqlite <filename>
```

If filename is missing it will use default database name: atpdatabase.db

### Example
```sql
select lastName from player join ranking on player.id = ranking.player_id where pos == 1 group by lastName;
```
