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
    columns = [
        "city", "price", "sqft_living", "sqft_lot", "sqft_above", "sqft_basement",
        "bedrooms", "bathrooms", "floors", "view", "condition", "waterfront",
        "has_basement", "was_renovated", "house_age", "date", "price_per_sqft",
    ]
    sales = pd.read_csv(path, usecols=columns).dropna(subset=["city", "price", "sqft_living"])
    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    return sales


def show_market_analysis():
    st.title("Market Analysis")
    st.caption("Explore the historical sales used in the EDA.")
    try:
        sales = load_market_data()
    except (OSError, ValueError):
        st.error("The cleaned housing dataset could not be loaded.")
        return

    cities = st.multiselect("Filter by city", sorted(sales["city"].unique()))
    visible = sales[sales["city"].isin(cities)] if cities else sales
    if visible.empty:
        st.info("No sales in this selection.")
        return

    def chart(figure):
        figure.update_layout(margin=dict(t=24, b=24))
        st.plotly_chart(figure)

    overview, prices, features, location, segments, time = st.tabs([
        "1. Overview", "2. Prices", "3. Features", "4. Location",
        "5. Segments", "6. Time",
    ])
    price = visible["price"]
    median = price.median()
    mean = price.mean()
    dates = visible["date"].dropna()
    with overview:
        st.subheader("What does a typical sale look like?")
        a, b, c = st.columns(3)
        a.metric("Sales in view", f"{len(visible):,}")
        b.metric("Median sale price", f"${median:,.0f}")
        c.metric("Median living area", f"{visible.sqft_living.median():,.0f} sq ft")
        st.write(f"{visible.city.nunique()} cities. Sales recorded from "
                 f"{dates.min():%b %d, %Y} to {dates.max():%b %d, %Y}." if len(dates)
                 else "No recorded sale dates in this selection.")
        beds = visible["bedrooms"].mode().iloc[0]
        baths = visible["bathrooms"].mode().iloc[0]
        st.write(f"The most common counts are {beds:g} bedrooms and {baths:g} bathrooms.")

    with prices:
        st.subheader("How spread out are sale prices?")
        use_log = st.toggle("Show log-transformed prices")
        values = np.log1p(price) if use_log else price
        title = "log(1 + sale price)" if use_log else "Sale price ($)"
        fig = px.histogram(x=values, nbins=30, labels={"x": title})
        markers = [("Mean", mean, "#E58A80"), ("Median", median, "#36B99C")]
        for label, amount, color in markers:
            position = np.log1p(amount) if use_log else amount
            fig.add_vline(x=position, line_color=color, line_dash="dash",
                          annotation_text=f"{label} ${amount:,.0f}",
                          annotation_position="top left" if label == "Mean" else "top right")
        chart(fig)
        q1, q3 = price.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((price < q1 - 1.5 * iqr) | (price > q3 + 1.5 * iqr)).sum()
        st.write(f"The mean is ${mean:,.0f} and the median is ${median:,.0f}. "
                 f"{outliers:,} sales fall outside the 1.5× IQR fences.")
        if use_log:
            st.caption("The reference lines show the raw-dollar mean and median at their transformed positions.")

    with features:
        st.subheader("Which property features move with price?")
        chart(px.scatter(visible, x="sqft_living", y="price", hover_name="city",
                         opacity=0.45, labels={"sqft_living": "Living area (sq ft)", "price": "Sale price ($)"}))
        feature_names = {
            "sqft_living": "Living area", "sqft_lot": "Lot size",
            "sqft_above": "Above-ground area", "sqft_basement": "Basement area",
            "bedrooms": "Bedrooms", "bathrooms": "Bathrooms", "floors": "Floors",
            "view": "View", "condition": "Condition", "waterfront": "Waterfront",
            "has_basement": "Basement", "was_renovated": "Renovated", "house_age": "House age",
        }
        correlations = visible[list(feature_names)].corrwith(price).dropna()
        if not correlations.empty:
            strongest = correlations.abs().idxmax()
            ranked = correlations.rename(index=feature_names).sort_values()
            chart(px.bar(x=ranked.values, y=ranked.index, orientation="h",
                         labels={"x": "Pearson correlation with sale price", "y": "Feature"}))
            st.write(f"{feature_names[strongest]} has the strongest linear association "
                     f"in this selection (r = {correlations[strongest]:+.2f}).")
        else:
            st.caption("Correlations require variation in price and property features.")
        options = ["bathrooms", "view", "waterfront", "has_basement"]
        feature = st.selectbox("Compare median price by", options, format_func=feature_names.get)
        groups = visible.groupby(feature)["price"].agg(median="median", sales="size").reset_index()
        groups["Group"] = (
            groups[feature].map({0: "No", 1: "Yes"})
            if feature in {"waterfront", "has_basement"} else groups[feature].astype(str)
        )
        chart(px.bar(groups, x="Group", y="median", hover_data=["sales"],
                     labels={"median": "Median sale price ($)"}))
        highest = groups.loc[groups["median"].idxmax()]
        st.caption(f"Highest group median in this view: {highest['Group']} at "
                   f"${highest['median']:,.0f} ({highest['sales']:,} sales). Group comparisons are descriptive.")

    with location:
        st.subheader("How do cities compare?")
        metrics = {"Median sale price": "price", "Median price per sq ft": "price_per_sqft"}
        label = st.radio("City ranking metric", list(metrics), horizontal=True)
        source = metrics[label]
        by_city = visible.groupby("city")[source].agg(median="median", sales="count").reset_index()
        qualified = by_city[by_city["sales"] >= 10].sort_values("median", ascending=False)
        if qualified.empty:
            st.info("No cities in this selection have 10 sales for the ranking.")
        else:
            ranking = st.radio("Show cities", ["Highest 10", "Lowest 10"], horizontal=True)
            shown = qualified.head(10) if ranking == "Highest 10" else qualified.tail(10)
            chart(px.bar(shown.sort_values("median"), x="median", y="city", orientation="h",
                         hover_data=["sales"], labels={"median": label + " ($)", "city": "City"}))
            leader = qualified.iloc[0] if ranking == "Highest 10" else qualified.iloc[-1]
            st.write(f"{leader['city']} ranks first in this view at ${leader['median']:,.0f} "
                     f"({leader['sales']:,} sales). Rankings include cities with at least 10 sales.")

    with segments:
        st.subheader("How do homes differ across price tiers?")
        low, high = sales["price"].quantile([0.25, 0.75])
        tiers = ["Low-cost", "Mid-market", "Premium"]
        segment = np.select([price <= low, price <= high], tiers[:2], default=tiers[2])
        summary = visible.assign(segment=segment).groupby("segment").agg(
            sales=("price", "size"), living_area=("sqft_living", "median"),
            bathrooms=("bathrooms", "median"), price_per_sqft=("price_per_sqft", "median"),
        ).reindex(tiers).dropna(how="all")
        st.write(f"Full-dataset cutoffs are ${low:,.0f} and ${high:,.0f}. "
                 "These cutoffs stay fixed when you filter by city.")
        st.dataframe(summary.rename(columns={
            "sales": "Sales", "living_area": "Median living area (sq ft)",
            "bathrooms": "Median bathrooms", "price_per_sqft": "Median price per sq ft ($)",
        }).round(1))
        largest = summary["living_area"].idxmax()
        st.caption(f"{largest} has the largest median living area among the displayed tiers "
                   f"({summary.loc[largest, 'living_area']:,.0f} sq ft).")

    with time:
        st.subheader("How did activity change over the recorded months?")
        dated = visible.dropna(subset=["date"])
        if dated.empty:
            st.info("No recorded sale dates in this selection.")
        else:
            monthly = dated.groupby(dated["date"].dt.to_period("M"))["price"].agg(
                sales="size", median="median").reset_index()
            monthly["month"] = monthly["date"].astype(str)
            for column, label in [("sales", "Sales"), ("median", "Median sale price ($)")]:
                chart(px.line(monthly, x="month", y=column, markers=True,
                              labels={"month": "Month", column: label}))
            busiest = monthly.loc[monthly["sales"].idxmax()]
            st.caption(f"{busiest['month']} had the most recorded sales in this selection "
                       f"({busiest['sales']:,}). First and last months may be partial.")

    st.subheader("From EDA to prediction")
    st.write("The EDA compares past sales across size, features, and location. The Ridge "
             "model uses property inputs such as living area and ZIP code to predict on "
             "a log-price scale. Open Price Predictor to see an estimate and its feature breakdown.")


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