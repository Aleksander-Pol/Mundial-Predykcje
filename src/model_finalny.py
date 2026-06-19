import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize_scalar
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PRZEWAGA_GOSPODARZA = 100

def waga_turnieju(t):
    if t == "FIFA World Cup": return 60
    if t in ("UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup"): return 50
    if "qualification" in t: return 40
    if t == "Friendly": return 20
    return 30

def indeks_goli(r):
    r = abs(r)
    if r <= 1: return 1
    if r == 2: return 1.5
    return (11 + r) / 8

# wczytanie danych
df = pd.read_csv(BASE / "data" / "results.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.dropna(subset=["home_score", "away_score"]).sort_values("date")

# przejście Elo: ratingi + dane treningowe (per mecz)
RATINGI = {}
diffs, gd, ga = [], [], []
for _, row in df.iterrows():
    gosp, gosc = row["home_team"], row["away_team"]
    Rg, Rs = RATINGI.get(gosp, 1500), RATINGI.get(gosc, 1500)
    bonus = 0 if row["neutral"] else PRZEWAGA_GOSPODARZA
    diff = (Rg + bonus) - Rs
    diffs.append(diff); gd.append(row["home_score"]); ga.append(row["away_score"])
    Ea = 1 / (1 + 10 ** (-diff / 400))
    Sa = 1 if row["home_score"] > row["away_score"] else (0.5 if row["home_score"] == row["away_score"] else 0)
    zmiana = waga_turnieju(row["tournament"]) * indeks_goli(row["home_score"] - row["away_score"]) * (Sa - Ea)
    RATINGI[gosp] = Rg + zmiana
    RATINGI[gosc] = Rs - zmiana

diffs = np.array(diffs); gd = np.array(gd).astype(int); ga = np.array(ga).astype(int)

# mostek Elo -> gole
B, A = np.polyfit(np.concatenate([diffs, -diffs]), np.concatenate([gd, ga]), 1)

# dopasowanie rho (Dixon-Coles)
lam = np.maximum(A + B * diffs, 0.05)
mu  = np.maximum(A - B * diffs, 0.05)
base = poisson.pmf(gd, lam) * poisson.pmf(ga, mu)
def _neg_ll(rho):
    t = np.ones_like(base)
    m = (gd==0)&(ga==0); t[m] = 1 - lam[m]*mu[m]*rho
    m = (gd==0)&(ga==1); t[m] = 1 + lam[m]*rho
    m = (gd==1)&(ga==0); t[m] = 1 + mu[m]*rho
    m = (gd==1)&(ga==1); t[m] = 1 - rho
    return -np.sum(np.log(np.maximum(base*t, 1e-15)))
RHO = minimize_scalar(_neg_ll, bounds=(-0.2, 0.2), method="bounded").x

# lista drużyn do dropdownów (min. 30 meczów)
def lista_druzyn(min_meczow=30):
    licznik = pd.concat([df["home_team"], df["away_team"]]).value_counts()
    return sorted(licznik[licznik >= min_meczow].index)

# predykcja
def przewiduj_mecz(team_a, team_b, neutralne=True, max_goli=8):
    bonus = 0 if neutralne else PRZEWAGA_GOSPODARZA
    diff = (RATINGI.get(team_a, 1500) + bonus) - RATINGI.get(team_b, 1500)
    lam_, mu_ = max(A + B*diff, 0.05), max(A - B*diff, 0.05)
    m = np.outer(poisson.pmf(np.arange(max_goli+1), lam_),
                 poisson.pmf(np.arange(max_goli+1), mu_))
    m[0,0] *= 1 - lam_*mu_*RHO
    m[0,1] *= 1 + lam_*RHO
    m[1,0] *= 1 + mu_*RHO
    m[1,1] *= 1 - RHO
    m /= m.sum()
    p1, px, p2 = np.tril(m,-1).sum(), np.trace(m), np.triu(m,1).sum()
    i, j = np.unravel_index(m.argmax(), m.shape)
    return {"wynik": f"{i}:{j}", "wygrana_a": p1, "remis": px, "wygrana_b": p2}