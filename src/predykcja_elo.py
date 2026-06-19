import numpy as np
from scipy.stats import poisson
from src.elo import wczytaj_dane, policz_elo, PRZEWAGA_GOSPODARZA

# policz ratingi i mostek raz, na pełnych danych
_df = wczytaj_dane()
RATINGI, _trening = policz_elo(_df)
B, A = np.polyfit(_trening["diff"], _trening["gole"], 1)   # gole ≈ A + B*diff

def rating(t):
    return RATINGI.get(t, 1500)

def przewiduj_mecz(team_a, team_b, neutralne=True, max_goli=8):
    bonus = 0 if neutralne else PRZEWAGA_GOSPODARZA
    diff = (rating(team_a) + bonus) - rating(team_b)

    lam_a = max(A + B * diff, 0.05)     # oczekiwane gole A
    lam_b = max(A - B * diff, 0.05)     # B ma lustrzaną różnicę

    m = np.outer(poisson.pmf(np.arange(max_goli + 1), lam_a),
                 poisson.pmf(np.arange(max_goli + 1), lam_b))
    p1, px, p2 = np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()
    i, j = np.unravel_index(m.argmax(), m.shape)

    return {"wynik": f"{i}:{j}", "wygrana_a": p1, "remis": px, "wygrana_b": p2}


if __name__ == "__main__":
    print("USA-AUS (USA u siebie):", przewiduj_mecz("United States", "Australia", neutralne=False))
    print("Brazil-Croatia:        ", przewiduj_mecz("Brazil", "Croatia"))