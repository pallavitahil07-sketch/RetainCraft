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

    urgency_labels = []
    churn_risk_pct = None

    if model is not None and n_rows > 0:
        try:
            preds = model.predict(df_work)
            pred_list = [str(p).strip().lower() for p in preds]
            for p in pred_list:
                if "high" in p or p in {"2", "urgent", "critical"}:
                    urgency_labels.append("High Urgency")
                elif "risk" in p or "med" in p or p in {"1", "at_risk", "at risk"}:
                    urgency_labels.append("At Risk")
                else:
                    urgency_labels.append("Low Risk")
            if len(urgency_labels) != n_rows:
                urgency_labels = []
        except Exception:
            urgency_labels = []

        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(df_work)
                max_probs = np.max(probs, axis=1)
                churn_risk_pct = np.clip(np.round(max_probs * 100, 1), 1.0, 99.0)
            except Exception:
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

    if churn_risk_pct is None or len(churn_risk_pct) != n_rows:
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


ROBOT_CART_SVG = """
<svg viewBox="0 0 360 220" width="100%" height="215" xmlns="http://www.w3.org/2000/svg">
  <g fill="#facc15">
    <polygon points="168,26 171,33 179,33 173,38 175,45 168,41 161,45 163,38 157,33 165,33"/>
    <polygon points="188,26 191,33 199,33 193,38 195,45 188,41 181,45 183,38 177,33 185,33"/>
    <polygon points="208,26 211,33 219,33 213,38 215,45 208,41 201,45 203,38 197,33 205,33"/>
    <polygon points="228,26 231,33 239,33 233,38 235,45 228,41 221,45 223,38 217,33 225,33"/>
    <polygon points="248,26 251,33 259,33 253,38 255,45 248,41 241,45 243,38 237,33 245,33"/>
  </g>
  <rect x="115" y="72" width="66" height="44" rx="12" fill="#9ca3af" opacity="0.85"/>
  <polygon points="162,116 174,116 172,128" fill="#9ca3af" opacity="0.85"/>
  <circle cx="138" cy="91" r="3" fill="#1e1b4b"/>
  <circle cx="156" cy="91" r="3" fill="#1e1b4b"/>
  <path d="M140 103 Q147 97 154 103" stroke="#1e1b4b" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <line x1="220" y1="54" x2="220" y2="72" stroke="#c4b5fd" stroke-width="3"/>
  <circle cx="220" cy="52" r="5" fill="#38bdf8"/>
  <rect x="188" y="72" width="64" height="54" rx="18" fill="#e0e7ff"/>
  <rect x="196" y="82" width="48" height="32" rx="10" fill="#1e1b4b"/>
  <circle cx="210" cy="96" r="4.5" fill="#38bdf8"/>
  <circle cx="230" cy="96" r="4.5" fill="#38bdf8"/>
  <path d="M213 105 Q220 110 227 105" stroke="#38bdf8" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <rect x="196" y="130" width="48" height="28" rx="10" fill="#c7d2fe"/>
  <path d="M172 146 L252 146 L242 178 L184 178 Z" fill="#6366f1" stroke="#a5b4fc" stroke-width="2"/>
  <circle cx="192" cy="188" r="7" fill="#38bdf8"/>
  <circle cx="234" cy="188" r="7" fill="#38bdf8"/>
  <rect x="268" y="138" width="10" height="36" rx="3" fill="#38bdf8"/>
  <rect x="283" y="118" width="10" height="56" rx="3" fill="#60a5fa"/>
  <rect x="298" y="96" width="10" height="78" rx="3" fill="#a78bfa"/>
  <polyline points="266,112 286,92 308,68" fill="none" stroke="#38bdf8" stroke-width="3" stroke-linecap="round"/>
  <polygon points="302,66 314,64 311,76" fill="#38bdf8"/>
</svg>
"""

if "scored_table" not in st.session_state:
    st.session_state.scored_table = None
if "scored_summary" not in st.session_state:
    st.session_state.scored_summary = None
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Home"

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 18% 12%, #3b1478 0%, #16092e 45%, #090414 100%);
        color: #f5f3ff;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    button[data-testid="stBaseButton-secondary"],
    .stButton > button[kind="secondary"] {
        background: rgba(45, 22, 92, 0.85) !important;
        color: #f5f3ff !important;
        border: 1px solid rgba(168, 85, 247, 0.45) !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    button[data-testid="stBaseButton-secondary"]:hover,
    .stButton > button[kind="secondary"]:hover {
        background: rgba(124, 58, 237, 0.9) !important;
        color: #ffffff !important;
        border-color: #c084fc !important;
    }
    button[data-testid="stBaseButton-primary"],
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #7c3aed 0%, #9333ea 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(216, 180, 254, 0.4) !important;
        border-radius: 999px !important;
        font-weight: 700 !important;
    }
    button[data-testid="stBaseButton-primary"]:hover,
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #8b5cf6 0%, #a855f7 100%) !important;
        color: #ffffff !important;
    }
    [data-testid="stTextArea"] textarea {
        background-color: rgba(18, 9, 40, 0.92) !important;
        color: #f5f3ff !important;
        border: 1px solid rgba(168, 85, 247, 0.45) !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(24, 12, 54, 0.88) !important;
        color: #f5f3ff !important;
        border: 1px dashed rgba(192, 132, 252, 0.55) !important;
        border-radius: 14px !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: rgba(88, 28, 135, 0.9) !important;
        color: #ffffff !important;
        border: 1px solid rgba(216, 180, 254, 0.4) !important;
        border-radius: 8px !important;
    }
    .top-brand {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f5f3ff;
        padding-top: 6px;
    }
    .right-badge {
        text-align: right;
        font-size: 0.85rem;
        color: #c4b5fd;
        padding-top: 8px;
    }
    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        line-height: 1.15;
        color: #ffffff;
        margin-bottom: 10px;
    }
    .hero-sub {
        font-size: 1rem;
        color: #d8b4fe;
        margin-bottom: 18px;
    }
    .panel-card {
        background: rgba(24, 12, 52, 0.82);
        border: 1px solid rgba(168, 85, 247, 0.35);
        border-radius: 18px;
        padding: 20px;
        margin-top: 12px;
    }
    .res-high {
        background: linear-gradient(145deg, rgba(127, 29, 29, 0.65), rgba(49, 16, 42, 0.85));
        border: 1px solid rgba(248, 113, 113, 0.45);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .res-risk {
        background: linear-gradient(145deg, rgba(88, 28, 135, 0.65), rgba(36, 15, 66, 0.85));
        border: 1px solid rgba(192, 132, 252, 0.45);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .res-low {
        background: linear-gradient(145deg, rgba(19, 78, 74, 0.65), rgba(15, 36, 56, 0.85));
        border: 1px solid rgba(45, 212, 191, 0.45);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .res-neutral {
        background: rgba(35, 18, 74, 0.82);
        border: 1px solid rgba(167, 139, 250, 0.35);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .res-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #f3e8ff;
        margin-bottom: 6px;
    }
    .res-num {
        font-size: 1.75rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.15;
    }
    .res-caption {
        font-size: 0.78rem;
        color: #c4b5fd;
        margin-top: 5px;
    }
    .sugg-box {
        background: rgba(25, 12, 56, 0.82);
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 14px;
        padding: 16px;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

b_col, n1, n2, n3, n4, r_badge = st.columns([1.6, 1.0, 1.0, 1.3, 1.0, 2.2])
with b_col:
    st.markdown('<div class="top-brand">✦ RetainCraft</div>', unsafe_allow_html=True)
with n1:
    if st.button("Home", use_container_width=True):
        st.session_state.active_tab = "Home"
with n2:
    if st.button("About", use_container_width=True):
        st.session_state.active_tab = "About"
with n3:
    if st.button("How It Works", use_container_width=True):
        st.session_state.active_tab = "How It Works"
with n4:
    if st.button("Insights", use_container_width=True):
        st.session_state.active_tab = "Insights"
with r_badge:
    st.markdown('<div class="right-badge">✦ Built with ♥ using ML</div>', unsafe_allow_html=True)

st.markdown("---")

if st.session_state.active_tab == "About":
    st.subheader("About RetainCraft")
    st.write(
        "RetainCraft turns customer reviews and order context into prioritized retention queues—classifying "
        "High Urgency, At Risk, and Low Risk items while estimating at-risk revenue and goodwill credits."
    )
elif st.session_state.active_tab == "How It Works":
    st.subheader("How It Works")
    st.write(
        "1. Paste customer reviews (one per line) or upload a CSV file.\n"
        "2. Click **✦ Analyze Reviews** to score urgency, churn risk, and retention actions.\n"
        "3. Open **Insights** or scroll below **Retention & Action Suggestions** to inspect and download scored rows."
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
        st.info("No reviews analyzed yet. Go to Home, paste reviews or upload a CSV, and click Analyze Reviews.")
else:
    hero_left, hero_right = st.columns([1.4, 1.0], gap="large")
    with hero_left:
        st.markdown('<div class="hero-title">RetainCraft –<br>Review Triage & Retention Studio</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-sub">Turn customer reviews into actionable insights —<br>prioritize issues, reduce churn, and keep more customers.</div>',
            unsafe_allow_html=True,
        )
        cta1, cta2, _ = st.columns([1.2, 1.2, 1.6])
        with cta1:
            if st.button("✦ Try the Model", type="primary", use_container_width=True):
                st.session_state.active_tab = "Home"
        with cta2:
            if st.button("▸ Learn More", use_container_width=True):
                st.session_state.active_tab = "How It Works"
    with hero_right:
        st.markdown(ROBOT_CART_SVG, unsafe_allow_html=True)

    left_box, right_box = st.columns([1.05, 1.25], gap="large")

    with left_box:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("### ✦ Paste Your Reviews")
        st.caption("Add customer reviews or upload a CSV file")
        pasted_text = st.text_area(
            "Enter reviews here... (one per line)",
            height=165,
            placeholder="Enter reviews here... (one per line)",
            label_visibility="collapsed",
        )
        st.markdown("<div style='text-align:center; color:#c4b5fd; margin:8px 0;'>OR</div>", unsafe_allow_html=True)
        uploaded_csv = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_csv is not None:
            df_loaded = pd.read_csv(uploaded_csv)
            st.session_state.uploaded_df = df_loaded
            st.session_state.uploaded_name = uploaded_csv.name
            st.caption(f"Loaded {len(df_loaded)} rows from `{uploaded_csv.name}`. Click **✦ Analyze Reviews** to score.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right_box:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        analyze_clicked = st.button("✦ Analyze Reviews", type="primary", use_container_width=True)
        st.markdown("### Results", unsafe_allow_html=True)

        if analyze_clicked:
            if st.session_state.get("uploaded_df") is not None and not st.session_state.uploaded_df.empty:
                scored_df, summary = score_reviews_dataframe(st.session_state.uploaded_df)
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
            st.info("Paste at least one review line or upload a CSV file to see predictions.")

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
                <div class="res-high">
                    <div class="res-label">▲ High Urgency</div>
                    <div class="res-num">{fmt_int(high_val)}</div>
                    <div class="res-caption">Needs Attention</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f"""
                <div class="res-risk">
                    <div class="res-label">▣ At Risk</div>
                    <div class="res-num">{fmt_int(at_risk_val)}</div>
                    <div class="res-caption">Monitor Closely</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(
                f"""
                <div class="res-low">
                    <div class="res-label">✓ Low Risk</div>
                    <div class="res-num">{fmt_int(low_risk_val)}</div>
                    <div class="res-caption">All Good</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f"""
                <div class="res-neutral">
                    <div class="res-label">₹ At-Risk Revenue Value</div>
                    <div class="res-num">{fmt_inr(at_risk_rev_val)}</div>
                    <div class="res-caption">Order value across elevated-risk reviews</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="res-neutral">
                    <div class="res-label">◔ Average Churn Risk</div>
                    <div class="res-num">{fmt_pct(avg_churn_val)}</div>
                    <div class="res-caption">Mean churn likelihood in current batch</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sugg-box">', unsafe_allow_html=True)
        st.markdown("**✦ Retention & Action Suggestions**")
        if st.session_state.get("scored_summary"):
            st.write(
                f"- **Escalate {fmt_int(high_val)} High-Urgency ticket(s) first:** Route negative/high-urgency complaints to priority support promptly.\n"
                f"- **Protect {fmt_inr(at_risk_rev_val)} in At-Risk order value:** Target {fmt_int(at_risk_val)} elevated-risk accounts (avg churn risk {fmt_pct(avg_churn_val)}; goodwill pool {fmt_inr(total_goodwill_inr)}).\n"
                f"- **Audit recurring friction drivers:** Inspect delivery delays, damaged arrivals, and quality defects below or in the Insights tab."
            )
        else:
            st.caption("No retention actions generated yet. Submit reviews or upload a CSV and click Analyze Reviews.")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.scored_table is not None and not st.session_state.scored_table.empty:
            st.markdown("#### Scored Review Analysis")
            st.dataframe(st.session_state.scored_table.head(25), use_container_width=True)
            csv_bytes = st.session_state.scored_table.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Full Scored CSV",
                data=csv_bytes,
                file_name="retaincraft_scored_reviews.csv",
                mime="text/csv",
            )

        st.markdown("</div>", unsafe_allow_html=True)