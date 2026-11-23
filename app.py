import os
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import requests
import numpy as np

st.set_page_config(page_title="Real Estate AI", page_icon="🏠")
@st.cache_data
def load_market_data():
    path = Path(__file__).resolve().parent / "data" / "real_estate_cleaned.csv"
    columns = ["city", "price", "sqft_living"]
    return pd.read_csv(path, usecols=columns).dropna()


def show_market_analysis():
    st.title("Market Analysis")
    st.caption("Historical sales from the project's cleaned housing dataset.")
    try:
        sales = load_market_data()
    except (OSError, ValueError):
        st.error("The cleaned housing dataset could not be loaded.")
        return

    cities = st.multiselect("Filter by city", sorted(sales["city"].unique()))
    visible = sales[sales["city"].isin(cities)] if cities else sales
    count, median = st.columns(2)
    count.metric("Sales in view", f"{len(visible):,}")
    median.metric("Median sale price", f"${visible['price'].median():,.0f}")

    st.subheader("Sale price distribution")
    distribution = px.histogram(
        visible, x="price", labels={"price": "Sale price"}, nbins=40
    )
    distribution.update_xaxes(tickprefix="$", tickformat="~s")
    distribution.update_yaxes(title="Sales")
    distribution.update_traces(marker_color="#36B99C")
    distribution.update_layout(height=340, margin=dict(t=16, b=36))
    st.plotly_chart(distribution, config={"displayModeBar": False})

    st.subheader("Living area and sale price")
    area = px.scatter(
        visible,
        x="sqft_living",
        y="price",
        hover_name="city",
        labels={"sqft_living": "Living area (sq ft)", "price": "Sale price"},
        opacity=0.5,
    )
    area.update_yaxes(tickprefix="$", tickformat="~s")
    area.update_traces(marker_color="#36B99C")
    area.update_layout(height=375, margin=dict(t=16, b=36))
    st.plotly_chart(area, config={"displayModeBar": False})

    st.subheader("Median sale price by city")
    by_city = (
        visible.groupby("city", as_index=False)
        .agg(median_price=("price", "median"), sales=("price", "size"))
        .nlargest(12, "sales")
        .sort_values("median_price")
    )
    city_prices = px.bar(
        by_city,
        x="median_price",
        y="city",
        orientation="h",
        hover_data={"sales": True, "median_price": ":$,.0f"},
        labels={"median_price": "Median sale price", "city": "City", "sales": "Sales"},
    )
    city_prices.update_xaxes(tickprefix="$", tickformat="~s")
    city_prices.update_traces(marker_color="#36B99C")
    city_prices.update_layout(height=420, margin=dict(t=16, b=36))
    st.plotly_chart(city_prices, config={"displayModeBar": False})
    st.caption("Shows up to 12 cities with the most sales in the current selection.")


st.sidebar.header("Real Estate AI")
page = st.sidebar.radio("Go to", ["Price Predictor", "Market Analysis"])
if page == "Market Analysis":
    show_market_analysis()
    st.stop()


st.title("🏠 Real Estate Price Predictor")
st.write("Predict a house price and understand the model's prediction.")


@st.cache_resource
def load_model():
    return joblib.load("models/ridge_model.joblib")


model = load_model()

FRIENDLY_NAMES = {
    "sqft_living": "living area",
    "sqft_above": "above-ground area",
    "sqft_lot": "lot size",
    "sqft_basement": "basement size",
    "bedrooms": "number of bedrooms",
    "bathrooms": "number of bathrooms",
    "floors": "number of floors",
    "view": "view quality",
    "condition": "house condition",
    "waterfront": "waterfront location",
    "house_age": "house age",
    "has_basement": "having a basement",
    "was_renovated": "renovation status",
    "zipcode": "neighborhood",
}


def get_contributions(model, house: pd.DataFrame) -> pd.DataFrame:
    """Per-feature contribution in log-price space (coef * scaled value)."""
    pre = model.named_steps["pre"]
    ridge = model.named_steps["model"]

    Xt = pre.transform(house)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt).ravel()

    df = pd.DataFrame({
        "feature": pre.get_feature_names_out(),
        "contribution": Xt * ridge.coef_,
    })
    df["feature"] = (
        df["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace(r"^cat__zipcode_.*", "zipcode", regex=True)
    )
    df = df.groupby("feature", as_index=False)["contribution"].sum()
    df["abs"] = df["contribution"].abs()
    return df.sort_values("abs", ascending=False).drop(columns="abs")


def build_summary(contrib_df, top_n=5):
    """Factual summary written by code (no LLM, no invention)."""
    top = contrib_df.head(top_n)
    ups = [FRIENDLY_NAMES.get(f, f) for f in top[top["contribution"] > 0]["feature"]]
    downs = [FRIENDLY_NAMES.get(f, f) for f in top[top["contribution"] < 0]["feature"]]

    parts = []
    if ups:
        parts.append(f"**Raised the estimate:** {', '.join(ups)}.")
    if downs:
        parts.append(f"**Lowered the estimate:** {', '.join(downs)}.")
    parts.append("These effects are measured against an average house in the data "
                 "the model learned from. They show patterns in the data, "
                 "not judgments about the house.")
    return "\n\n".join(parts)


def describe_contributions(contrib_df, top_n=5):
    lines = []
    for r in contrib_df.head(top_n).itertuples():
        name = FRIENDLY_NAMES.get(r.feature, r.feature)
        direction = "raised the estimate" if r.contribution > 0 else "lowered the estimate"
        lines.append(f"- {name}: {direction}")
    return "\n".join(lines)


def build_prompt(price, top_contrib):
    drivers = describe_contributions(top_contrib)
    return f"""
Write a short, friendly explanation for a home buyer or seller with no technical background.

Estimated price: \\${price:,.0f}

What affected the estimate (compared to an average house in our data):
{drivers}

Write 3 short paragraphs:
1. State the estimated price.
2. Explain which factors raised the estimate and which lowered it, in plain words.
3. Remind the reader this is an automated estimate, not a guaranteed price or a professional appraisal.

Tone: warm, friendly and encouraging. You may use light expressions such as
"a nice boost" or "a slight drag", but ONLY about how a factor affected the estimate.

Strict rules:
- Use only the information above.
- Never make claims about the condition, quality or desirability of the house or neighborhood.
- Never say the house has flaws or problems.
- Do not explain WHY a factor had its effect; only say whether it raised or lowered the estimate.
- Do not mention any numbers except the price.
"""


st.subheader("Enter House Details")

with st.form("house_form"):
    sqft_living = st.number_input("Living area (sqft)", min_value=100, value=1500)
    sqft_lot = st.number_input("Lot area (sqft)", min_value=100, value=5000)
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=15.0, value=2.0, step=0.25)
    floors = st.number_input("Floors", min_value=1.0, max_value=4.0, value=1.0, step=0.5)
    view = st.selectbox("View quality (0–4)", [0, 1, 2, 3, 4])
    condition = st.selectbox("House condition (1–5)", [1, 2, 3, 4, 5], index=2)
    waterfront = st.selectbox("Waterfront", ["No", "Yes"]) == "Yes"
    sqft_above = st.number_input("Above-ground area (sqft)", min_value=0, value=1200)
    sqft_basement = st.number_input("Basement area (sqft, 0 if none)", min_value=0, value=300)
    house_age = st.number_input("House age (years)", min_value=0, value=20)
    was_renovated = st.selectbox("Was renovated?", ["No", "Yes"]) == "Yes"
    zipcode = st.text_input("Zipcode", value="98118")

    submitted = st.form_submit_button("Predict Price")

if submitted:
    if sqft_above + sqft_basement != sqft_living:
        st.warning("Heads up: above-ground area + basement area doesn't equal "
                   "the living area. Please double-check your numbers.")

    raw = {
        "sqft_living": sqft_living, "sqft_lot": sqft_lot, "bedrooms": bedrooms,
        "bathrooms": bathrooms, "floors": floors, "view": view,
        "condition": condition, "waterfront": int(waterfront),
        "sqft_above": sqft_above, "sqft_basement": sqft_basement,
        "house_age": house_age,
        "has_basement": int(sqft_basement > 0),   # computed, not typed
        "was_renovated": int(was_renovated),
        "zipcode": str(zipcode),
    }

    model_input = raw.copy()
    for c in ["sqft_living", "sqft_lot", "sqft_above", "sqft_basement"]:
        model_input[c] = np.log1p(raw[c])
    house = pd.DataFrame([model_input])

    predicted_price = float(np.expm1(model.predict(house)[0]))
    contributions = get_contributions(model, house)

    st.session_state["result"] = {
        "price": predicted_price,
        "contributions": contributions,
    }
    st.session_state.pop("explanation", None)

if "result" in st.session_state:
    res = st.session_state["result"]

    st.subheader("Prediction")
    st.metric("Estimated house price", f"${res['price']:,.0f}")

    low, high = res["price"] * 0.85, res["price"] * 1.15
    st.caption(f"Typical range: \\${low:,.0f} – \\${high:,.0f} "
           "(the model's typical error is around 11%, so the real price can differ).")

    st.subheader("What drove this estimate?")
    top = res["contributions"].head(8).copy()
    top["feature"] = top["feature"].map(FRIENDLY_NAMES).fillna(top["feature"])
    top["direction"] = "No effect"
    top.loc[top["contribution"] > 0, "direction"] = "Raised estimate"
    top.loc[top["contribution"] < 0, "direction"] = "Lowered estimate"
    top["value_label"] = top["contribution"].map("{:+.3f}".format)

    fig = px.bar(
        top,
        x="contribution",
        y="feature",
        orientation="h",
        color="direction",
        text="value_label",
        color_discrete_map={
            "Raised estimate": "#36B99C",
            "Lowered estimate": "#E58A80",
            "No effect": "#9CA3AF",
        },
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        marker_line_width=0,
        hovertemplate="%{y}<br>%{fullData.name}: %{x:+.3f}<extra></extra>",
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=top["feature"].iloc[::-1].tolist(),
        title=None,
        fixedrange=True,
        automargin=True,
    )
    span = max(top["contribution"].abs().max(), 0.01)
    fig.update_xaxes(
        range=[
            min(0, top["contribution"].min()) - 0.22 * span,
            max(0, top["contribution"].max()) + 0.22 * span,
        ],
        visible=False,
        fixedrange=True,
    )
    fig.add_vline(x=0, line_width=1, line_color="#88919D", layer="below")
    fig.update_layout(
        height=330,
        margin=dict(l=4, r=12, t=8, b=8),
        bargap=0.42,
        showlegend=False,
    )
    st.plotly_chart(fig, config={"displayModeBar": False})
    st.caption("Positive values raised the estimate. Negative values lowered it. "
               "Values are contributions on the model's log-price scale.")

    st.subheader("Summary")
    st.write(build_summary(res["contributions"]))

    if st.button("Explain Prediction with AI"):
        prompt = build_prompt(res["price"], res["contributions"])
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                try:
                    api_key = st.secrets.get("GROQ_API_KEY")
                except FileNotFoundError:
                    api_key = None

            messages = [{"role": "user", "content": prompt}]
            with st.spinner("Generating explanation..."):
                if api_key:
                    response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={
                            "model": "openai/gpt-oss-20b",
                            "messages": messages,
                            "reasoning_effort": "low",
                            "max_completion_tokens": 800,
                        },
                        timeout=30,
                    )
                    response.raise_for_status()
                    explanation = response.json()["choices"][0]["message"]["content"]
                else:
                    response = requests.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": "llama3.2:3b",
                            "messages": messages,
                            "stream": False,
                            "options": {"temperature": 0.2, "num_predict": 350},
                        },
                        timeout=30,
                    )
                    response.raise_for_status()
                    explanation = response.json()["message"]["content"]

                if not isinstance(explanation, str) or not explanation.strip():
                    raise ValueError("Empty explanation")
                st.session_state["explanation"] = explanation.strip()
        except (requests.RequestException, KeyError, IndexError, ValueError, TypeError):
            st.error("AI explanation is temporarily unavailable. "
                     "Check the Groq key or local Ollama and try again.")

    if "explanation" in st.session_state:
        st.subheader("AI Explanation")
        st.write(st.session_state["explanation"])
        st.caption("AI-written text. It may be imperfect; the Summary above is "
                   "generated directly from the model's numbers.")