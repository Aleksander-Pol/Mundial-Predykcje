import pandas as pd
import numpy as np
from scipy.stats import poisson
from collections import Counter

# --- wczytanie danych z szerokim oknem historii ---
df = pd.read_csv("../data/results.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.dropna(subset=["home_score", "away_score"])
df = df[df["date"] >= "2000-01-01"]      # długa historia do treningu

# --- podział po czasie: trening = starsze, test = nowsze ---
podzial = "2024-06-01"
train = df[df["date"] < podzial]
test  = df[df["date"] >= podzial]

# --- policz siły ataku/obrony TYLKO na danych treningowych ---
def policz_sily(mecze):
    home = mecze[["home_team", "away_team", "home_score"]].rename(
        columns={"home_team": "team", "away_team": "opponent", "home_score": "goals"})
    away = mecze[["away_team", "home_team", "away_score"]].rename(
        columns={"away_team": "team", "home_team": "opponent", "away_score": "goals"})
    dl = pd.concat([home, away])
    srednia = dl["goals"].mean()
    atak = dl.groupby("team")["goals"].mean() / srednia
    obrona = dl.groupby("opponent")["goals"].mean() / srednia
    return srednia, atak, obrona

srednia, atak, obrona = policz_sily(train)

# --- typ pojedynczego meczu: "1" (gospodarz), "X" (remis), "2" (gość) ---
def typ_meczu(a, b, max_goli=8):
    lam_a = srednia * atak[a] * obrona[b]
    lam_b = srednia * atak[b] * obrona[a]
    m = np.outer(poisson.pmf(np.arange(max_goli + 1), lam_a),
                 poisson.pmf(np.arange(max_goli + 1), lam_b))
    p1, px, p2 = np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()
    return ["1", "X", "2"][np.argmax([p1, px, p2])]

# --- przejście po meczach testowych i porównanie z rzeczywistością ---
typy = Counter()
prawdy = Counter()
trafione, liczba = 0, 0
for _, row in test.iterrows():
    a, b = row["home_team"], row["away_team"]
    if a not in atak.index or b not in atak.index:
        continue  # drużyna bez historii w treningu — pomijamy
    typ = typ_meczu(a, b)
    if row["home_score"] > row["away_score"]:
        prawda = "1"
    elif row["home_score"] == row["away_score"]:
        prawda = "X"
    else:
        prawda = "2"
    typy[typ] += 1
    prawdy[prawda] += 1
    trafione += (typ == prawda)
    liczba += 1

print(f"Sprawdzono meczów: {liczba}")
print(f"Trafność: {trafione / liczba:.1%}")
print("Model typował:   ", dict(typy))
print("Naprawdę było:   ", dict(prawdy))