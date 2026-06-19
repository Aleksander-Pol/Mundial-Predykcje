import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize_scalar
from collections import Counter
from src.elo import wczytaj_dane, waga_turnieju, indeks_goli, PRZEWAGA_GOSPODARZA

df = wczytaj_dane()
podzial = pd.Timestamp("2024-06-01")

# pomocnicza: liczy diff SPRZED meczu i aktualizuje ratingi
def aktualizuj(ratingi, row):
    gosp, gosc = row["home_team"], row["away_team"]
    Rg, Rs = ratingi.get(gosp, 1500), ratingi.get(gosc, 1500)
    bonus = 0 if row["neutral"] else PRZEWAGA_GOSPODARZA
    diff = (Rg + bonus) - Rs
    Ea = 1 / (1 + 10 ** (-diff / 400))
    Sa = 1 if row["home_score"] > row["away_score"] else (0.5 if row["home_score"] == row["away_score"] else 0)
    zmiana = waga_turnieju(row["tournament"]) * indeks_goli(row["home_score"] - row["away_score"]) * (Sa - Ea)
    ratingi[gosp] = Rg + zmiana
    ratingi[gosc] = Rs - zmiana
    return diff

# === PRZEBIEG 1: zbierz dane treningowe (point-in-time), dopasuj mostek i rho ===
ratingi = {}
diffs, gd, ga = [], [], []
for _, row in df[df["date"] < podzial].iterrows():
    diffs.append(aktualizuj(ratingi, row))
    gd.append(row["home_score"]); ga.append(row["away_score"])
diffs, gd, ga = np.array(diffs), np.array(gd).astype(int), np.array(ga).astype(int)

# mostek Elo -> gole
B, A = np.polyfit(np.concatenate([diffs, -diffs]), np.concatenate([gd, ga]), 1)

# dopasowanie rho metodą największej wiarogodności
lam = np.maximum(A + B * diffs, 0.05)
mu  = np.maximum(A - B * diffs, 0.05)
base = poisson.pmf(gd, lam) * poisson.pmf(ga, mu)      # niezależny Poisson

def neg_log_wiarogodnosc(rho):
    t = np.ones_like(base)
    m = (gd == 0) & (ga == 0); t[m] = 1 - lam[m] * mu[m] * rho
    m = (gd == 0) & (ga == 1); t[m] = 1 + lam[m] * rho
    m = (gd == 1) & (ga == 0); t[m] = 1 + mu[m] * rho
    m = (gd == 1) & (ga == 1); t[m] = 1 - rho
    return -np.sum(np.log(np.maximum(base * t, 1e-15)))

RHO = minimize_scalar(neg_log_wiarogodnosc, bounds=(-0.2, 0.2), method="bounded").x
print("Dopasowane rho:", round(RHO, 4))

# === funkcja prawdopodobieństw z korektą Dixon-Coles ===
def prawdopodobienstwa(diff, max_goli=8):
    lam_, mu_ = max(A + B * diff, 0.05), max(A - B * diff, 0.05)
    m = np.outer(poisson.pmf(np.arange(max_goli + 1), lam_),
                 poisson.pmf(np.arange(max_goli + 1), mu_))
    m[0, 0] *= 1 - lam_ * mu_ * RHO     # korekta 4 niskich wyników
    m[0, 1] *= 1 + lam_ * RHO
    m[1, 0] *= 1 + mu_ * RHO
    m[1, 1] *= 1 - RHO
    m /= m.sum()                        # renormalizacja do sumy 1
    return np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()

# === PRZEBIEG 2: zbuduj ratingi od nowa i oceń mecze testowe ===
ratingi = {}
typy, prawdy = Counter(), Counter()
trafione, liczba, log_loss_suma = 0, 0, 0.0
for _, row in df.iterrows():
    diff = aktualizuj(ratingi, row)
    if row["date"] >= podzial:
        p1, px, p2 = prawdopodobienstwa(diff)
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