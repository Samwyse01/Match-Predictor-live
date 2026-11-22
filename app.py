import pandas as pd
import streamlit as st

st.set_page_config(page_title="Match Predictor", layout="wide")
st.title("⚽ Match predictor")
st.caption("Model estimates vs bookmaker odds. For information only, not betting advice. Predictions can be wrong.")


st.sidebar.header("Futbol24")
st.sidebar.link_button("Live scores", "https://www.futbol24.com/")
st.sidebar.link_button("Compare teams (H2H)", "https://www.futbol24.com/team-compare")
st.sidebar.caption("Opens futbol24.com so you can cross-check fixtures and results.")


def load(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


tab1, tab2 = st.tabs(["Match result (1X2)", "Corners & shots"])

with tab1:
    df = load("docs/predictions.csv")
    if df is None:
        st.warning("No predictions yet. Run `python -m src.predict` first.")
    else:
        league = st.multiselect("League", sorted(df.league.unique()), default=list(df.league.unique()))
        only_value = st.checkbox("Only show value flags")
        view = df[df.league.isin(league)]
        if only_value and "value_flag" in view:
            view = view[view.value_flag == True]  # noqa: E712
        st.dataframe(view, use_container_width=True, hide_index=True)

with tab2:
    tot = load("docs/totals_predictions.csv")
    if tot is None:
        st.warning("No totals yet. Run `python -m src.predict --totals`.")
    else:
        stat = st.selectbox("Market", sorted(tot.stat.unique()))
        v = tot[tot.stat == stat]
        st.caption("Fair odds = 1 / probability. A bet only has value if the bookmaker's odds are HIGHER than the fair odds.")
        st.dataframe(v, use_container_width=True, hide_index=True)
