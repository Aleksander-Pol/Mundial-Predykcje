import numpy as np
import pandas as pd
from collections import Counter
from src.elo import (wczytaj_dane, policz_elo, dopasuj_mostek, dopasuj_rho,
                     prawdopodobienstwa, aktualizuj)

df = wczytaj_dane()
podzial = pd.Timestamp("2024-06-01")

# === trening: tylko mecze sprzed daty podziału ===
train = df[df["date"] < podzial]
_, diffs, gole_home, gole_away = policz_elo(train)
B, A = dopasuj_mostek(diffs, gole_home, gole_away)
RHO = dopasuj_rho(diffs, gole_home, gole_away, A, B)

# === test: zbuduj ratingi od nowa i oceń mecze testowe ===
ratingi = {}
typy, prawdy = Counter(), Counter()
trafione, liczba, log_loss_suma = 0, 0, 0.0
for _, row in df.iterrows():
    diff = aktualizuj(ratingi, row)
    if row["date"] >= podzial:
        p1, px, p2 = prawdopodobienstwa(diff, A, B, RHO)
        typ = ["1", "X", "2"][np.argmax([p1, px, p2])]
        prawda = "1" if row["home_score"] > row["away_score"] else ("X" if row["home_score"] == row["away_score"] else "2")
        log_loss_suma += -np.log(max({"1": p1, "X": px, "2": p2}[prawda], 1e-15))
        typy[typ] += 1; prawdy[prawda] += 1
        trafione += (typ == prawda); liczba += 1

print(f"Sprawdzono meczów: {liczba}")
print(f"Trafność: {trafione / liczba:.1%}")
print(f"Log-loss: {log_loss_suma / liczba:.4f}")
print("Model typował:", dict(typy))
print("Naprawdę było:", dict(prawdy))