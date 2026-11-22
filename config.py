from datetime import date

# football-data.co.uk league codes (free results + bookmaker odds + match stats)
LEAGUES = {
    "E0": "Premier League",
    "E1": "Championship",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

LABELS = ["H", "D", "A"]  # home win, draw, away win
EDGE_THRESHOLD = 0.05      # model prob must beat bookmaker no-vig prob by this much
MIN_GAMES = 5              # teams need this many prior games for a reliable row

# Head-to-head: recent meetings count more (exponential decay), shrunk toward "no information"
H2H_HALF_LIFE_DAYS = 730   # a meeting 2 years ago counts half as much as one today
H2H_MAX_MEETINGS = 10
H2H_PRIOR = 2.0            # pseudo-weight pulling H2H toward neutral when few meetings exist

# Totals markets: (home column, away column) in football-data.co.uk files
STAT_COLS = {
    "corners": ("HC", "AC"),
    "shots": ("HS", "AS"),
    "shots_on_target": ("HST", "AST"),
}
DEFAULT_TOTAL = {"corners": 10.5, "shots": 24.0, "shots_on_target": 8.5}  # prior for unknown teams
STAT_WINDOW = 10           # last N games used for team corner/shot averages
LINES = {
    "corners": [8.5, 9.5, 10.5, 11.5],
    "shots": [22.5, 24.5, 26.5],
    "shots_on_target": [7.5, 8.5, 9.5],
}


def season_codes(n: int = 5, today: date | None = None) -> list[str]:
    """Return the last n season codes like '2526' (season starts in August)."""
    today = today or date.today()
    start = today.year if today.month >= 8 else today.year - 1
    return [f"{(y % 100):02d}{((y + 1) % 100):02d}" for y in range(start - n + 1, start + 1)]
