import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize_scalar
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent  # to pobiera ścieżkę do tego pliku w którym jestem tj. elo.py

PRZEWAGA_GOSPODARZA = 100  # zmienna arbitralna jak bardzo lepiej idzie gospodarzom jak grają u siebie


def wczytaj_dane():
    df = pd.read_csv(BASE / "data" / "results.csv")
    df["date"] = pd.to_datetime(df["date"])  # zamieniamy kolumnę z datą z tekstowej na format daty dostarczany przez pandas do porównywania itd
    df = df.dropna(subset=["home_score", "away_score"]).sort_values("date")  # zachowujemy tylko te mecze które nie mają wyniku jako null
    return df


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


# liczy diff SPRZED meczu i aktualizuje ratingi
def aktualizuj(ratingi, row):
    gosp, gosc = row["home_team"], row["away_team"]
    Rg, Rs = ratingi.get(gosp, 1500), ratingi.get(gosc, 1500)
    bonus = 0 if row["neutral"] else PRZEWAGA_GOSPODARZA
    diff = (Rg + bonus) - Rs  # mierzymy różnice ELO pomiędzy gospodarzem a gościem z uwzględnieniem bonusu gospodarza
    expected_score_a = 1 / (1 + 10 ** (-diff / 400))
    actual_score_a = 1 if row["home_score"] > row["away_score"] else (0.5 if row["home_score"] == row["away_score"] else 0)
    zmiana = waga_turnieju(row["tournament"]) * indeks_goli(row["home_score"] - row["away_score"]) * (actual_score_a - expected_score_a)
    """
    Tutaj podobnie do perceptronu to wygląda że stała uczenia to u nas waga_turnieju*indeks_goli
    """
    ratingi[gosp] = Rg + zmiana
    ratingi[gosc] = Rs - zmiana
    return diff


# przejście chronologiczne: ratingi + dane treningowe (per mecz, point-in-time)
def policz_elo(mecze):
    ratingi = {}
    diffs, gole_home, gole_away = [], [], []
    for _, row in mecze.iterrows():
        diffs.append(aktualizuj(ratingi, row))
        gole_home.append(row["home_score"])
        gole_away.append(row["away_score"])
    diffs = np.array(diffs)
    gole_home = np.array(gole_home).astype(int)
    gole_away = np.array(gole_away).astype(int)
    return ratingi, diffs, gole_home, gole_away


# mostek Elo -> gole
def dopasuj_mostek(diffs, gole_home, gole_away):
    B, A = np.polyfit(np.concatenate([diffs, -diffs]), np.concatenate([gole_home, gole_away]), 1)
    """
    Dopasowanie funkcji liniowej do rozrzuconych punktów na wykresie. Polyfit obsługuje nie tylko proste lecz
    n-wymiarowe wielomiany. Ostatni parametr właśnie wskazuje na stopień wielomianu
    """
    return B, A


# dopasowanie rho (Dixon-Coles)
def dopasuj_rho(diffs, gole_home, gole_away, A, B):
    lam = np.maximum(A + B * diffs, 0.05)  # tablica wszystkich lambd ze sportkań
    mu  = np.maximum(A - B * diffs, 0.05)  # analogicznie
    base = poisson.pmf(gole_home, lam) * poisson.pmf(gole_away, mu)  # tablica wzysztkich prawdopodobncych wyników na zakończenie spotkania zgodnie z tym jak naprawde było

    def _neg_ll(rho):
        t = np.ones_like(base)  # tworzy nową tablicę o takim samym kształcie jak tablica w parametrze
        m = (gole_home == 0) & (gole_away == 0); t[m] = 1 - lam[m] * mu[m] * rho
        m = (gole_home == 0) & (gole_away == 1); t[m] = 1 + lam[m] * rho
        m = (gole_home == 1) & (gole_away == 0); t[m] = 1 + mu[m] * rho
        m = (gole_home == 1) & (gole_away == 1); t[m] = 1 - rho
        """
        Dixon-Coles ze wzoru z neta
        """
        return -np.sum(np.log(np.maximum(base * t, 1e-15)))  # dodawanie stabilniejsze niż iloczyn. działanie w log-space jest bezpieczniejsze

    return minimize_scalar(_neg_ll, bounds=(-0.2, 0.2), method="bounded").x  # minimalizacja funkcji żeby znaleźć namniejsze RHO


# macierz wyników z korektą Dixon-Coles
def macierz_wynikow(diff, A, B, RHO, max_goli=8):
    lam_, mu_ = max(A + B * diff, 0.05), max(A - B * diff, 0.05)
    m = np.outer(poisson.pmf(np.arange(max_goli + 1), lam_),
                 poisson.pmf(np.arange(max_goli + 1), mu_))
    m[0, 0] *= 1 - lam_ * mu_ * RHO
    m[0, 1] *= 1 + lam_ * RHO
    m[1, 0] *= 1 + mu_ * RHO
    m[1, 1] *= 1 - RHO
    m /= m.sum()  # normalizacja
    return m


# prawdopodobieństwa 1 / X / 2
def prawdopodobienstwa(diff, A, B, RHO, max_goli=8):
    m = macierz_wynikow(diff, A, B, RHO, max_goli)
    return np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()