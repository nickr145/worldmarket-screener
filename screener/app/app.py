"""Local Streamlit dashboard: streamlit run screener/app/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from screener.app import services as S  # noqa: E402
from screener.app.views import detail, explainer, modelcard, screen  # noqa: E402

st.set_page_config(page_title="Market screener", layout="wide")
st.warning(
    "**Screening tool: describes recent momentum and volatility in listed-company data, not causal advice or a forecast.**  \n"
    f"**Data snapshot: {S.SNAPSHOT_DATE}** (latest price {S.LATEST_PRICE_DATE}) · Decision date {S.DECISION_DATE} · model version {S.MODEL_VERSION} (frozen) · "
    f"Low-skill screen: out-of-sample rank correlation about {S.SKILL_GROWTH} for growth and {S.SKILL_DOWNSIDE} for downside volatility (2023-2024 only); "
    "every label is a rough guide.",
    icon="⚠️",
)
PAGES = {"Screen": screen.render, "Market detail": detail.render, "Label explainer": explainer.render, "Model card": modelcard.render}
st.sidebar.title("Market screener")
page = st.sidebar.radio("Page", list(PAGES), key="page")
st.sidebar.caption(f"Data snapshot {S.SNAPSHOT_DATE}  \n{S.MODEL_VERSION} · decision date {S.DECISION_DATE}")
PAGES[page]()
