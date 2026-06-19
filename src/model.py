# średnia liczba goli, jaką drużyna strzela w pojedynczym meczu (cały zbiór)
from src.dane import long_df



# ile meczów rozegrała każda drużyna w naszym oknie czasowym
liczba_meczow = long_df.groupby("team").size()

# uznajemy tylko "prawdziwe" reprezentacje: min. 30 meczów od 2018
uznane = liczba_meczow[liczba_meczow >= 30].index

# zostawiamy mecze, w których OBIE drużyny są uznane
long_df = long_df[long_df["team"].isin(uznane) & long_df["opponent"].isin(uznane)]


srednia = long_df["goals"].mean()

# siła ataku: ile średnio strzela KAŻDA drużyna, podzielone przez średnią ogólną
atak = long_df.groupby("team")["goals"].mean() / srednia

# siła obrony: ile średnio TRACI każda drużyna, podzielone przez średnią ogólną
# trik: grupujemy po 'opponent' — bo gole strzelone PRZECIWKO drużynie to jej gole stracone
obrona = long_df.groupby("opponent")["goals"].mean() / srednia


def oczekiwane_gole(druzyna_a, druzyna_b):
    # ile goli średnio strzeli A przeciwko B
    lam_a = srednia * atak[druzyna_a] * obrona[druzyna_b]
    # ile goli średnio strzeli B przeciwko A
    lam_b = srednia * atak[druzyna_b] * obrona[druzyna_a]
    return lam_a, lam_b


if __name__ == "__main__":
    print(oczekiwane_gole("Brazil", "Croatia"))