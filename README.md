# Match Predictor

Estimates win/draw/loss probabilities for football matches and compares them with bookmaker odds
(the same kind of 1X2 odds SportyBet shows). It is a data/ML project, **not** a money machine:
bookmaker odds already contain most public information plus a margin. Predictions can be wrong.
Never stake money you can't afford to lose, and check the betting laws where you live.

## How it works
- `src/data.py`: free results + odds from football-data.co.uk (no scraping of betting sites)
- `src/features.py`: Elo rating, last-5 form, goals for/against, rest days, **time-decayed head-to-head** (recent meetings count more, a 2-year-old meeting counts half; shrunk toward neutral when meetings are few), rolling corner/shot averages. No look-ahead leakage.
- `src/model.py`: logistic regression (default) or gradient boosting for 1X2; Poisson regression + negative binomial for total corners / shots / shots on target
- `src/backtest.py`: trains on the past, tests on the future, compares log loss with bookmaker odds
- `src/predict.py`: predicts upcoming fixtures, flags "value" where model prob beats no-vig bookmaker prob by 5%+
- `app.py`: Streamlit viewer; `.github/workflows/predict.yml`: daily run that commits `docs/predictions.csv`

## Run
```bash
pip install -r requirements.txt
python -m src.backtest --leagues E0 --seasons 6     # does the model beat the bookmaker? (it usually won't)
python -m src.predict --leagues E0 SP1 D1 I1 F1     # writes docs/predictions.csv
python -m src.predict --totals                       # also writes docs/totals_predictions.csv (corners, shots, shots on target)
python -m src.predict --no-h2h                       # switch head-to-head off
streamlit run app.py
python -m src.backtest --totals                      # also backtest corners/shots; prints with-vs-without H2H
python -m src.backtest --demo --totals               # offline test with synthetic data
pytest -q
```

## Reading the backtest
Lower log loss is better. If `model_logloss` is not below `bookmaker_logloss`, the model has no edge
and any "value" flags are noise. Judge ROI only over thousands of bets, never a few hundred.

## Corners and shots
football-data.co.uk has no odds for these markets, so the output gives the expected total, P(over)
for common lines (e.g. corners 8.5-11.5) and **fair odds** (1 / probability). Compare them with what
your bookmaker offers: a bet only has value if the offered odds are higher than the fair odds, and
only if the backtest shows the totals model beating `mae_league_mean` and `over_logloss_baseline`.

## Adding Futbol24 (or any other source)
The code does not scrape futbol24.com. Bring your own CSV export (or a feed you are licensed to use)
and merge it in:
```bash
python -m src.predict --extra-results futbol24_results.csv --extra-fixtures futbol24_fixtures.csv \
       --aliases data/team_aliases.csv --col-map data/futbol24_columns.json
python -m src.backtest --extra-results futbol24_results.csv --aliases data/team_aliases.csv --col-map data/futbol24_columns.json
```
- `--col-map`: JSON mapping our names to your column names, e.g. `{"HomeTeam": "Home", "AwayTeam": "Away", "FTHG": "HG", "FTAG": "AG"}`
- `--aliases`: CSV with `alias,canonical` so "Man Utd" and "Man United" are the same club. The run prints team names it could not match.
- On duplicate matches, football-data.co.uk wins. Corner/shot columns (`HC,AC,HS,AS,HST,AST`) are used if present.
- Check Futbol24's terms before any automated access.

The Streamlit sidebar also has buttons that open Futbol24's live scores and team-compare pages for cross-checking.
