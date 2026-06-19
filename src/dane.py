import pandas as pd

from pathlib import Path

# katalog główny projektu: dane.py jest w src/, więc cofamy się o jeden poziom
BASE = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE / "data" / "results.csv")

# zamiana kolumny 'date' z tekstu na prawdziwy typ daty
df["date"] = pd.to_datetime(df["date"])

# zostawiamy tylko mecze od 2018-01-01 włącznie
df = df[df["date"] >= "2018-01-01"]

# dla pewności usuwamy mecze bez wyniku
df = df.dropna(subset=["home_score", "away_score"])

# perspektywa gospodarza
home = pd.DataFrame({
    "team": df["home_team"],
    "opponent": df["away_team"],
    "goals": df["home_score"],
    "home": (~df["neutral"]).astype(int),   # 1 gdy mecz NIE na neutralnym boisku
})

# perspektywa gościa
away = pd.DataFrame({
    "team": df["away_team"],
    "opponent": df["home_team"],
    "goals": df["away_score"],
    "home": 0,   # gość nigdy nie ma przewagi gospodarza
})

# sklejamy obie perspektywy w jedną ramkę
long_df = pd.concat([home, away], ignore_index=True)

