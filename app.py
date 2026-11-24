from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="RetainCraft – Review Triage & Retention Studio",
    page_icon="✦",
    layout="wide",
)

DEFAULT_PRICE = 1499.0
DEFAULT_DAYS = 3.0


def fmt_int(val):
    if isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool):
        if pd.isna(val):
            return "—"
        return f"{int(round(val)):,}"
    return "—" if val in (None, "", "—") else str(val)


def fmt_inr(val):
    if isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool):
        if pd.isna(val):
            return "—"
        return f"₹{int(round(val)):,}"
    return "—" if val in (None, "", "—") else str(val)


def fmt_pct(val):
    if isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool):
        if pd.isna(val):
            return "—"
        return f"{float(val):.1f}%"
    return "—" if val in (None, "", "—") else str(val)


@st.cache_resource
def load_triage_model():
    model_path = Path("retaincraft_triage_model.joblib")
    if model_path.exists():
        return joblib.load(model_path)
    return None


def score_reviews_dataframe(df_in: pd.DataFrame):
    df_work = df_in.copy()

    review_col = next(
        (c for c in ["review_text", "review", "text", "comment", "feedback"] if c in df_work.columns),
        None,
    )
    if review_col is None:
        df_work["review_text"] = ""
    elif review_col != "review_text":
        df_work["review_text"] = df_work[review_col]

    df_work["review_text"] = df_work["review_text"].fillna("").astype(str)

    if "review_id" not in df_work.columns:
        df_work["review_id"] = [f"REV-{i + 1:04d}" for i in range(len(df_work))]

    price_col = next(
        (c for c in ["price_inr", "order_value", "price", "amount"] if c in df_work.columns),
        None,
    )
    raw_prices = (
        df_work[price_col]
        if price_col is not None
        else pd.Series(DEFAULT_PRICE, index=df_work.index, dtype="float64")
    )
    prices_series = pd.to_numeric(raw_prices, errors="coerce").fillna(DEFAULT_PRICE)

    days_col = next(
        (c for c in ["days_since_delivery", "delivery_days", "days"] if c in df_work.columns),
        None,
    )
    raw_days = (
        df_work[days_col]
        if days_col is not None
        else pd.Series(DEFAULT_DAYS, index=df_work.index, dtype="float64")
    )
    days_series = pd.to_numeric(raw_days, errors="coerce").fillna(DEFAULT_DAYS)

    cat_col = next((c for c in ["category", "product_category"] if c in df_work.columns), None)
    category_series = (
        df_work[cat_col].fillna("General").astype(str)
        if cat_col is not None
        else pd.Series("General", index=df_work.index)
    )

    city_col = next((c for c in ["city", "location"] if c in df_work.columns), None)
    city_series = (
        df_work[city_col].fillna("Bengaluru").astype(str)
        if city_col is not None
        else pd.Series("Bengaluru", index=df_work.index)
    )

    df_work["price_inr"] = prices_series
    df_work["days_since_delivery"] = days_series
    df_work["category"] = category_series
    df_work["city"] = city_series

    model = load_triage_model()
    n_rows = len(df_work)

    if model is not None and n_rows > 0:
        try:
            preds = model.predict(df_work)
            pred_list = [str(p).strip().lower() for p in preds]
        except Exception:
            pred_list = []

        urgency_labels = []
        for idx, p in enumerate(pred_list):
            if "high" in p or p in {"2", "urgent", "critical"}:
                urgency_labels.append("High Urgency")
            elif "risk" in p or "med" in p or p in {"1", "at_risk", "at risk"}:
                urgency_labels.append("At Risk")
            else:
                urgency_labels.append("Low Risk")

        if len(urgency_labels) != n_rows:
            urgency_labels = []

        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(df_work)
                max_probs = np.max(probs, axis=1)
                churn_risk_pct = np.clip(np.round(max_probs * 100, 1), 1.0, 99.0)
            except Exception:
                churn_risk_pct = None
        else:
            churn_risk_pct = None
    else:
        urgency_labels = []
        churn_risk_pct = None

    if not urgency_labels:
        urgent_words = (
            "broken", "damaged", "late", "delay", "refund", "fake", "counterfeit",
            "heat", "burnt", "worst", "missing", "leaking", "horrible", "refused",
        )
        risk_words = (
            "dull", "tight", "average", "flimsy", "missing", "delay", "okay", "dented", "slow",
        )
        urgency_labels = []
        for txt, d_days in zip(df_work["review_text"].str.lower(), df_work["days_since_delivery"]):
            if any(w in txt for w in urgent_words) or d_days >= 8:
                urgency_labels.append("High Urgency")
            elif any(w in txt for w in risk_words) or d_days >= 4:
                urgency_labels.append("At Risk")
            else:
                urgency_labels.append("Low Risk")

    if churn_risk_pct is None:
        base_map = {"High Urgency": 68.0, "At Risk": 36.0, "Low Risk": 9.5}
        churn_risk_pct = np.array([base_map[lbl] for lbl in urgency_labels], dtype="float64")

    goodwill_inr = []
    for lbl, price in zip(urgency_labels, df_work["price_inr"]):
        if lbl == "High Urgency":
            goodwill_inr.append(round(min(max(price * 0.20, 150.0), 1500.0)))
        elif lbl == "At Risk":
            goodwill_inr.append(round(min(max(price * 0.10, 75.0), 750.0)))
        else:
            goodwill_inr.append(0)

    df_work["urgency_class"] = urgency_labels
    df_work["churn_risk_pct"] = churn_risk_pct
    df_work["goodwill_credit_inr"] = goodwill_inr

    high_count = int(sum(1 for x in urgency_labels if x == "High Urgency"))
    at_risk_count = int(sum(1 for x in urgency_labels if x == "At Risk"))
    low_count = int(sum(1 for x in urgency_labels if x == "Low Risk"))

    elevated_mask = df_work["urgency_class"].isin(["High Urgency", "At Risk"])
    at_risk_revenue_inr = float(df_work.loc[elevated_mask, "price_inr"].sum())
    avg_churn_pct = float(np.mean(churn_risk_pct)) if n_rows > 0 else 0.0
    total_goodwill_inr = float(df_work["goodwill_credit_inr"].sum())

    summary = {
        "high_urgency": high_count,
        "at_risk": at_risk_count,
        "low_risk": low_count,
        "at_risk_revenue_inr": at_risk_revenue_inr,
        "avg_churn_pct": avg_churn_pct,
        "total_goodwill_inr": total_goodwill_inr,
    }
    return df_work, summary


if "scored_table" not in st.session_state:
    st.session_state.scored_table = None
if "scored_summary" not in st.session_state:
    st.session_state.scored_summary = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Home"

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #241046 0%, #0d081c 65%);
        color: #f5f3ff;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(167, 139, 250, 0.25);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .res-label {
        font-size: 0.85rem;
        color: #c4b5fd;
        margin-bottom: 6px;
    }
    .res-num {
        font-size: 1.75rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }
    .res-caption {
        font-size: 0.8rem;
        color: #a78bfa;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

nav1, nav2, nav3, nav4, _ = st.columns([1.2, 1.2, 1.4, 1.2, 4])
with nav1:
    if st.button("Home", use_container_width=True):
        st.session_state.active_tab = "Home"
with nav2:
    if st.button("About", use_container_width=True):
        st.session_state.active_tab = "About"
with nav3:
    if st.button("How It Works", use_container_width=True):
        st.session_state.active_tab = "How It Works"
with nav4:
    if st.button("Insights", use_container_width=True):
        st.session_state.active_tab = "Insights"

st.title("RetainCraft – Review Triage & Retention Studio")
st.caption("Turn customer reviews into prioritized retention actions — reduce churn and protect revenue.")

if st.session_state.active_tab == "About":
    st.subheader("About RetainCraft")
    st.write(
        "RetainCraft triages incoming customer reviews and order context into High Urgency, "
        "At Risk, and Low Risk queues while estimating at-risk revenue and goodwill retention credits."
    )
elif st.session_state.active_tab == "How It Works":
    st.subheader("How It Works")
    st.write(
        "1. Paste customer reviews (one per line) or upload a CSV file.\n"
        "2. Click **✦ Analyze Reviews** to score urgency, churn risk, and retention actions.\n"
        "3. Review prioritized queues and export results from the Insights tab."
    )
elif st.session_state.active_tab == "Insights":
    st.subheader("Scored Queue & Retention Insights")
    if st.session_state.scored_table is not None and not st.session_state.scored_table.empty:
        st.dataframe(st.session_state.scored_table, use_container_width=True)
        csv_bytes = st.session_state.scored_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Scored Reviews CSV",
            data=csv_bytes,
            file_name="retaincraft_scored_reviews.csv",
            mime="text/csv",
        )
    else:
        st.info("No analyzed dataset yet. Go to Home, add reviews or upload a CSV, and click Analyze Reviews.")
else:
    left_box, right_box = st.columns([1.1, 1.2], gap="large")

    with left_box:
        st.subheader("✦ Paste Your Reviews")
        pasted_text = st.text_area(
            "Enter reviews here... (one per line)",
            height=170,
            placeholder="Enter reviews here... (one per line)",
        )
        st.markdown("**OR**")
        uploaded_csv = st.file_uploader("Upload CSV File", type=["csv"])

    with right_box:
        analyze_clicked = st.button("✦ Analyze Reviews", type="primary", use_container_width=True)
        st.markdown('<div class="res-label" style="margin-top:14px;">Results</div>', unsafe_allow_html=True)

        if analyze_clicked:
            if uploaded_csv is not None:
                df_csv = pd.read_csv(uploaded_csv)
                scored_df, summary = score_reviews_dataframe(df_csv)
                st.session_state.scored_table = scored_df
                st.session_state.scored_summary = summary
            else:
                lines = [line.strip() for line in pasted_text.splitlines() if line.strip()]
                if lines:
                    df_input = pd.DataFrame({"review_text": lines})
                    scored_df, summary = score_reviews_dataframe(df_input)
                    st.session_state.scored_table = scored_df
                    st.session_state.scored_summary = summary
                else:
                    st.session_state.scored_table = None
                    st.session_state.scored_summary = None
                    st.warning("Paste at least one review line or upload a CSV file to see predictions.")
        elif not st.session_state.get("scored_summary"):
            st.info("Paste at least one review line or upload a CSV file, then click Analyze Reviews.")

        summary = st.session_state.get("scored_summary") or {}
        high_val = summary.get("high_urgency", "—")
        at_risk_val = summary.get("at_risk", "—")
        low_risk_val = summary.get("low_risk", "—")
        at_risk_rev_val = summary.get("at_risk_revenue_inr", "—")
        avg_churn_val = summary.get("avg_churn_pct", "—")
        total_goodwill_val = summary.get("total_goodwill_inr", "—")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="res-label">High Urgency</div>
                    <div class="res-num">{fmt_int(high_val)}</div>
                    <div class="res-caption">Needs Attention</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="res-label">At Risk</div>
                    <div class="res-num">{fmt_int(at_risk_val)}</div>
                    <div class="res-caption">Monitor Closely</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="res-label">Low Risk</div>
                    <div class="res-num">{fmt_int(low_risk_val)}</div>
                    <div class="res-caption">All Good</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="res-label">At-Risk Revenue Value</div>
                    <div class="res-num">{fmt_inr(at_risk_rev_val)}</div>
                    <div class="res-caption">Elevated-risk orders</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="res-label">Average Churn Risk</div>
                    <div class="res-num">{fmt_pct(avg_churn_val)}</div>
                    <div class="res-caption">Mean batch churn risk</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="res-label">Goodwill Budget</div>
                    <div class="res-num">{fmt_inr(total_goodwill_val)}</div>
                    <div class="res-caption">Suggested credits</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Retention & Action Suggestions")
        if st.session_state.get("scored_summary"):
            st.write(
                f"- **Escalate {fmt_int(high_val)} High-Urgency ticket(s) first:** Route critical complaints to priority support immediately.\n"
                f"- **Protect {fmt_inr(at_risk_rev_val)} in At-Risk order value:** Follow up with {fmt_int(at_risk_val)} at-risk customers (avg churn {fmt_pct(avg_churn_val)}; estimated goodwill pool {fmt_inr(total_goodwill_inr)}).\n"
                f"- **Audit top friction drivers:** Check delivery delays, damaged arrivals, and product quality trends in the Insights tab."
            )
        else:
            st.caption("Run an analysis to view retention recommendations.")