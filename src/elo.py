import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

def wczytaj_dane():
    df = pd.read_csv(BASE / "data" / "results.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    return df.sort_values("date")

PRZEWAGA_GOSPODARZA = 100

def waga_turnieju(t):
    if t == "FIFA World Cup":
        return 60
    if t in ("UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup"):
        return 50
    if "qualification" in t:
        return 40
    if t == "Friendly":
        return 20
    return 30

def indeks_goli(roznica):
    roznica = abs(roznica)
    if roznica <= 1:
        return 1
    if roznica == 2:
        return 1.5
    return (11 + roznica) / 8

def policz_elo(mecze):
    ratingi = {}
    dane = []   # pary (roznica_elo_przed_meczem, gole) z perspektywy obu drużyn

    def get_rating(t):
        return ratingi.get(t, 1500)

    for _, row in mecze.iterrows():
        a, b = row["home_team"], row["away_team"]
        Ra, Rb = get_rating(a), get_rating(b)

        bonus = 0 if row["neutral"] else PRZEWAGA_GOSPODARZA
        diff_a = (Ra + bonus) - Rb          # przewaga gospodarza w ratingu

        # ZAPIS PRZED AKTUALIZACJĄ = point-in-time, oba punkty widzenia
        dane.append((diff_a, row["home_score"]))
        dane.append((-diff_a, row["away_score"]))

        Ea = 1 / (1 + 10 ** (-diff_a / 400))
        if row["home_score"] > row["away_score"]:
            Sa = 1
        elif row["home_score"] == row["away_score"]:
            Sa = 0.5
        else:
            Sa = 0

        K = waga_turnieju(row["tournament"])
        G = indeks_goli(row["home_score"] - row["away_score"])
        zmiana = K * G * (Sa - Ea)
        ratingi[a] = Ra + zmiana
        ratingi[b] = Rb - zmiana

    return ratingi, pd.DataFrame(dane, columns=["diff", "gole"])


if __name__ == "__main__":
    df = wczytaj_dane()
    ratingi, trening = policz_elo(df)

    # dopasowanie mostka: gole ≈ a + b * roznica_elo
    b, a = np.polyfit(trening["diff"], trening["gole"], 1)
    print("a (gole przy równych drużynach):", round(a, 3))
    print("b (gole na 1 pkt Elo):", round(b, 6))
    print("→ przewaga 100 pkt Elo daje +", round(b * 100, 2), "gola")