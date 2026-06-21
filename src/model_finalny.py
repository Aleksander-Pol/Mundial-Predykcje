import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize_scalar
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent  #to pobiera ścieżkę do tego pliku w którym jestem tj. model finalny

PRZEWAGA_GOSPODARZA = 100 #zmienna arbitralna jak bardzo lepiej idzie gospodarzom jak grają u siebie

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
df["date"] = pd.to_datetime(df["date"]) # zamieniamy kolumnę z datą z tekstowej na format daty dostarczany przez pandas do porównywania itd
df = df.dropna(subset=["home_score", "away_score"]).sort_values("date") # zachowujemy tylko te mecze które nie mają wyniku jako null

# przejście Elo: ratingi + dane treningowe (per mecz)
RATINGI = {}
diffs, gole_home, gole_away = [], [], []
for _, row in df.iterrows():
    gosp, gosc = row["home_team"], row["away_team"]
    rating_gosp, rating_gosc = RATINGI.get(gosp, 1500), RATINGI.get(gosc, 1500)
    bonus = 0 if row["neutral"] else PRZEWAGA_GOSPODARZA
    diff = (rating_gosp + bonus) - rating_gosc  #mierzymy różnice ELO pomiędzy gospodarzem a gościem z uwzględnieniem bonusu gospodarza
    diffs.append(diff)
    gole_home.append(row["home_score"])
    gole_away.append(row["away_score"])

    expected_score_a = 1 / (1 + 10 ** (-diff / 400))
    actual_score_a = 1 if row["home_score"] > row["away_score"] else (0.5 if row["home_score"] == row["away_score"] else 0)

    zmiana = waga_turnieju(row["tournament"]) * indeks_goli(row["home_score"] - row["away_score"]) * (actual_score_a - expected_score_a)
    """
    Tutaj podobnie do perceptronu to wygląda że stała uczenia to u nas waga_turnieju*indeks_goli
    """
    RATINGI[gosp] = rating_gosp + zmiana
    RATINGI[gosc] = rating_gosc - zmiana

diffs = np.array(diffs)
gole_home = np.array(gole_home).astype(int)
gole_away = np.array(gole_away).astype(int)

# mostek Elo -> gole
B, A = np.polyfit(np.concatenate([diffs, -diffs]), np.concatenate([gole_home, gole_away]), 1)
"""
Dopasowanie funkcji liniowej do rozrzuconych punktów na wykresie. Polyfit obsługuje nie tylko proste lecz
n-wymiarowe wielomiany. Ostatni parametr właśnie wskazuje na stopień wielomianu
"""

# dopasowanie rho (Dixon-Coles)
lam = np.maximum(A + B * diffs, 0.05) #tablica wszystkich lambd ze sportkań
mu  = np.maximum(A - B * diffs, 0.05) #analogicznie
base = poisson.pmf(gole_home, lam) * poisson.pmf(gole_away, mu) #tablica wzysztkich prawdopodobncych wyników na zakończenie spotkania zgodnie z tym jak naprawde było


def _neg_ll(rho):
    t = np.ones_like(base) # tworzy nową tablicę o takim samym kształcie jak tablica w parametrze
    m = (gole_home == 0) & (gole_away == 0); t[m] = 1 - lam[m] * mu[m] * rho
    m = (gole_home == 0) & (gole_away == 1); t[m] = 1 + lam[m] * rho
    m = (gole_home == 1) & (gole_away == 0); t[m] = 1 + mu[m] * rho
    m = (gole_home == 1) & (gole_away == 1); t[m] = 1 - rho
    """
    Dixon-Coles ze wzoru z neta
    """

    return -np.sum(np.log(np.maximum(base * t, 1e-15))) # dodawanie stabilniejsze niż iloczyn. działanie w log-space jest bezpieczniejsze

RHO = minimize_scalar(_neg_ll, bounds=(-0.2, 0.2), method="bounded").x  # minimalizacja funkcji żeby znaleźć namniejsze RHO

# lista drużyn do dropdownów (min. 30 meczów)
def lista_druzyn(min_meczow=30):
    licznik = pd.concat([df["home_team"], df["away_team"]]).value_counts() #słownik z ilością wystąpień każdej z drużyn
    return sorted(licznik[licznik >= min_meczow].index) #maska na odsiewanie drużyn które mają za mało spotkań żeby móc robić jakieś predykcje

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
    m /= m.sum() # normalizacja


    p1, px, p2 = np.tril(m,-1).sum(), np.trace(m), np.triu(m,1).sum()
    i, j = np.unravel_index(m.argmax(), m.shape)
    return {"wynik": f"{i}:{j}", "wygrana_a": p1, "remis": px, "wygrana_b": p2}