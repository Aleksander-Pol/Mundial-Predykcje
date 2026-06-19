from src.model import oczekiwane_gole
import numpy as np
from scipy.stats import poisson


def przewiduj_mecz(druzyna_a, druzyna_b, max_goli=8):
    lam_a, lam_b = oczekiwane_gole(druzyna_a, druzyna_b)

    prob_a = poisson.pmf(np.arange(max_goli + 1), lam_a)
    prob_b = poisson.pmf(np.arange(max_goli + 1), lam_b)
    macierz = np.outer(prob_a, prob_b)

    wygrana_a = np.tril(macierz, -1).sum()
    remis     = np.trace(macierz)
    wygrana_b = np.triu(macierz, 1).sum()

    i, j = np.unravel_index(macierz.argmax(), macierz.shape)

    # zamiast print — zwracamy słownik z wynikami
    return {
        "wynik": f"{i}:{j}",
        "wygrana_a": wygrana_a,
        "remis": remis,
        "wygrana_b": wygrana_b,
    }


if __name__ == "__main__":
    print(przewiduj_mecz("Brazil", "Croatia"))


if __name__ == "__main__":
    przewiduj_mecz("United States", "Australia")