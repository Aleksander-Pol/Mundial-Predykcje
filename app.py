import streamlit as st
import plotly.graph_objects as go
from src.model_finalny import przewiduj_mecz, lista_druzyn

st.title("Predyktor meczów reprezentacji ⚽")

druzyny = lista_druzyn()

kol1, kol2 = st.columns(2)
with kol1:
    team_a = st.selectbox("Drużyna A", druzyny, index=druzyny.index("Brazil"))
with kol2:
    team_b = st.selectbox("Drużyna B", druzyny, index=druzyny.index("Croatia"))

neutralne = st.checkbox("Neutralne boisko (np. mundial)", value=True)

if st.button("Przewiduj"):
    w = przewiduj_mecz(team_a, team_b, neutralne=neutralne)

    st.subheader(f"Najbardziej prawdopodobny wynik:  {w['wynik']}")

    # pasek prawdopodobieństwa 1 / X / 2
    fig = go.Figure()
    segmenty = [
        (f"Wygrana {team_a}", w["wygrana_a"], "#2563eb"),
        ("Remis",             w["remis"],     "#9ca3af"),
        (f"Wygrana {team_b}", w["wygrana_b"], "#dc2626"),
    ]
    for nazwa, wartosc, kolor in segmenty:
        fig.add_trace(go.Bar(
            y=["Szanse"], x=[wartosc], name=nazwa, orientation="h",
            marker_color=kolor,
            text=f"{wartosc:.0%}", textposition="inside",
            insidetextanchor="middle",
            hovertemplate=f"{nazwa}: {wartosc:.1%}<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        height=200,
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis=dict(showticklabels=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)