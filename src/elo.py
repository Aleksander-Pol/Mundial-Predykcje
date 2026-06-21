import numpy as np
import pandas as pd
from src.elo import (wczytaj_dane, policz_elo, dopasuj_mostek, dopasuj_rho,
                     macierz_wynikow, PRZEWAGA_GOSPODARZA)

# trening na CAŁEJ historii (raz przy imporcie)
df = wczytaj_dane()
RATINGI, diffs, gole_home, gole_away = policz_elo(df)
B, A = dopasuj_mostek(diffs, gole_home, gole_away)
RHO = dopasuj_rho(diffs, gole_home, gole_away, A, B)


# lista drużyn do dropdownów (min. 30 meczów)
def lista_druzyn(min_meczow=30):
    licznik = pd.concat([df["home_team"], df["away_team"]]).value_counts()  # słownik z ilością wystąpień każdej z drużyn
    return sorted(licznik[licznik >= min_meczow].index)  # maska na odsiewanie drużyn które mają za mało spotkań żeby móc robić jakieś predykcje


# predykcja
def przewiduj_mecz(team_a, team_b, neutralne=True, max_goli=8):
    bonus = 0 if neutralne else PRZEWAGA_GOSPODARZA
    diff = (RATINGI.get(team_a, 1500) + bonus) - RATINGI.get(team_b, 1500)
    m = macierz_wynikow(diff, A, B, RHO, max_goli)
    p1, px, p2 = np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()
    i, j = np.unravel_index(m.argmax(), m.shape)
    return {"wynik": f"{i}:{j}", "wygrana_a": p1, "remis": px, "wygrana_b": p2}