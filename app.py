import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="RetainCraft — Review Triage & Retention Studio", layout="wide")

URGENCY_LABELS = {
    0: "Positive / Low Urgency",
    1: "Neutral / Medium Urgency",
    2: "Negative / High Urgency",
}

URGENCY_PALETTE = {
    "Negative / High Urgency": "#FF3366",
    "Neutral / Medium Urgency": "#00E5FF",
    "Positive / Low Urgency": "#34D399",
}

if "nav_tab" not in st.session_state:
    st.session_state.nav_tab = "Home"
if "pasted_reviews" not in st.session_state:
    st.session_state.pasted_reviews = ""
if "scored_summary" not in st.session_state:
    st.session_state.scored_summary = None
if "scored_table" not in st.session_state:
    st.session_state.scored_table = None

st.markdown(
    """
    <style>
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        .stApp {
            background: radial-gradient(circle at 78% 20%, rgba(139, 92, 246, 0.34), transparent 44%),
                        radial-gradient(circle at 18% 78%, rgba(79, 70, 229, 0.24), transparent 46%),
                        linear-gradient(180deg, #090716 0%, #120d2b 52%, #080613 100%) !important;
            color: #f8fafc !important;
        }
        .top-brand {
            font-size: 1.3rem;
            font-weight: 800;
            color: #ffffff;
            padding-top: 6px;
            letter-spacing: -0.01em;
        }
        div[data-testid="stButton"] > button[kind="secondary"] {
            background: rgba(30, 24, 66, 0.85) !important;
            color: #e2e8f0 !important;
            border: 1px solid rgba(167, 139, 250, 0.32) !important;
            border-radius: 999px !important;
            font-weight: 600 !important;
            padding: 0.45rem 1rem !important;
        }
        div[data-testid="stButton"] > button[kind="secondary"]:hover {
            background: rgba(109, 40, 217, 0.55) !important;
            color: #ffffff !important;
            border-color: rgba(196, 181, 253, 0.7) !important;
        }
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(90deg, #7c3aed 0%, #9333ea 50%, #a855f7 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(216, 180, 254, 0.45) !important;
            border-radius: 999px !important;
            font-weight: 700 !important;
            padding: 0.52rem 1.2rem !important;
            box-shadow: 0 8px 24px rgba(139, 92, 246, 0.38) !important;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            filter: brightness(1.08) !important;
            color: #ffffff !important;
        }
        div[data-testid="stTextArea"] textarea {
            background-color: #171236 !important;
            color: #f8fafc !important;
            border: 1px solid rgba(139, 92, 246, 0.35) !important;
            border-radius: 14px !important;
        }
        div[data-testid="stTextArea"] textarea::placeholder {
            color: #94a3b8 !important;
        }
        section[data-testid="stFileUploaderDropzone"] {
            background-color: #171236 !important;
            border: 1.5px dashed rgba(139, 92, 246, 0.48) !important;
            border-radius: 14px !important;
            color: #e2e8f0 !important;
        }
        section[data-testid="stFileUploaderDropzone"] * {
            color: #cbd5e1 !important;
        }
        section[data-testid="stFileUploaderDropzone"] button {
            background: rgba(109, 40, 217, 0.5) !important;
            color: #ffffff !important;
            border: 1px solid rgba(167, 139, 250, 0.45) !important;
            border-radius: 999px !important;
        }
        .main-hero-title {
            font-size: 2.85rem;
            font-weight: 800;
            line-height: 1.12;
            color: #ffffff;
            margin-bottom: 14px;
        }
        .main-hero-title span {
            color: #a78bfa;
        }
        .main-hero-sub {
            font-size: 1.05rem;
            line-height: 1.55;
            color: #cbd5e1;
            margin-bottom: 20px;
        }
        .card-title {
            font-size: 1.12rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 4px;
        }
        .card-sub {
            font-size: 0.88rem;
            color: #94a3b8;
            margin-bottom: 12px;
        }
        .or-divider {
            text-align: center;
            color: #94a3b8;
            font-size: 0.8rem;
            letter-spacing: 0.1em;
            margin: 10px 0;
        }
        .res-high {
            background: linear-gradient(135deg, rgba(190, 24, 93, 0.42), rgba(136, 19, 55, 0.30));
            border: 1px solid rgba(251, 113, 133, 0.48);
            border-radius: 14px;
            padding: 14px 16px;
        }
        .res-mid {
            background: linear-gradient(135deg, rgba(109, 40, 217, 0.46), rgba(76, 29, 149, 0.32));
            border: 1px solid rgba(196, 181, 253, 0.45);
            border-radius: 14px;
            padding: 14px 16px;
        }
        .res-low {
            background: linear-gradient(135deg, rgba(13, 148, 136, 0.40), rgba(19, 78, 74, 0.30));
            border: 1px solid rgba(45, 212, 191, 0.45);
            border-radius: 14px;
            padding: 14px 16px;
        }
        .val-revenue {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.24), rgba(6, 78, 59, 0.32));
            border: 1px solid rgba(52, 211, 153, 0.45);
            border-radius: 14px;
            padding: 14px 16px;
        }
        .val-churn {
            background: linear-gradient(135deg, rgba(244, 63, 94, 0.25), rgba(136, 19, 55, 0.32));
            border: 1px solid rgba(251, 113, 133, 0.45);
            border-radius: 14px;
            padding: 14px 16px;
        }
        .suggestions-box {
            background: rgba(23, 18, 54, 0.92);
            border: 1px solid rgba(167, 139, 250, 0.36);
            border-radius: 14px;
            padding: 14px 18px;
            margin-top: 12px;
        }
        .suggestions-title {
            font-size: 0.98rem;
            font-weight: 700;
            color: #e9d5ff;
            margin-bottom: 8px;
        }
        .suggestions-list {
            margin: 0;
            padding-left: 18px;
            color: #e2e8f0;
            font-size: 0.9rem;
            line-height: 1.55;
        }
        .res-label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 4px;
        }
        .res-num {
            font-size: 1.65rem;
            font-weight: 800;
            color: #ffffff;
            line-height: 1.15;
        }
        .res-caption {
            font-size: 0.8rem;
            color: #cbd5e1;
            margin-top: 4px;
        }
        .footer-pill {
            background: rgba(24, 19, 54, 0.82);
            border: 1px solid rgba(139, 92, 246, 0.24);
            border-radius: 12px;
            padding: 12px 14px;
            text-align: center;
            font-weight: 600;
            color: #f1f5f9;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_triage_pipeline():
    return joblib.load("retaincraft_triage_model.joblib")


def compute_risk_and_goodwill(probs_row, price_inr=1999.0, days_since_delivery=7):
    delay_ratio = float(np.clip(days_since_delivery / 14.0, 0.0, 1.0))
    raw_risk = 0.65 * float(probs_row[2]) + 0.20 * float(probs_row[1]) + 0.15 * delay_ratio
    churn_risk_pct = round(float(np.clip(raw_risk * 100.0, 1.0, 99.0)), 1)

    raw_credit = float(price_inr) * (churn_risk_pct / 100.0) * 0.15
    bounded_credit = float(np.clip(raw_credit, 50.0, 1000.0))
    goodwill_inr = int(round(bounded_credit / 10.0) * 10)
    return churn_risk_pct, goodwill_inr


def build_suggestions(summary):
    high_u = summary["high_urgency"]
    at_r = summary["at_risk"]
    avg_churn = summary["avg_churn_pct"]
    at_risk_rev = summary["at_risk_revenue_inr"]
    total_goodwill = summary["total_goodwill_inr"]

    tips = []
    if high_u > 0:
        tips.append(
            f"<b>Escalate {high_u:,} High-Urgency ticket(s) first:</b> Route negative/high-urgency complaints to priority support within 2 hours to prevent repeat churn."
        )
    if at_r > 0:
        tips.append(
            f"<b>Protect ₹{at_risk_rev:,} in At-Risk order value:</b> Target the {at_r:,} elevated-risk accounts (avg churn risk {avg_churn}%) with proactive check-ins and targeted goodwill credits (total pool ₹{total_goodwill:,})."
        )
    else:
        tips.append(
            f"<b>Maintain current retention pace:</b> Average churn risk is {avg_churn}% with no high-risk backlog—keep standard post-delivery follow-ups active."
        )
    tips.append(
        "<b>Audit recurring friction drivers:</b> Check top-ranked rows for repeated delivery delays, damaged arrivals, or quality defects to fix root causes upstream."
    )
    return tips


def score_reviews_dataframe(df_in, default_price=1999.0, default_days=7):
    text_col = "clean_review_text" if "clean_review_text" in df_in.columns else "review_text"
    prod_col = "product_label_short" if "product_label_short" in df_in.columns else "product_name"

    prices_series = pd.to_numeric(df_in.get("price_inr", default_price), errors="coerce").fillna(default_price)
    days_series = pd.to_numeric(df_in.get("days_since_delivery", default_days), errors="coerce").fillna(default_days)

    prepared = pd.DataFrame(
        {
            "clean_review_text": df_in[text_col].fillna("").astype(str),
            "price_inr": prices_series,
            "days_since_delivery": days_series,
            "product_label_short": (
                df_in[prod_col].fillna("Customer Order").astype(str).str.slice(0, 40)
                if prod_col in df_in.columns
                else pd.Series(["Customer Order"] * len(df_in))
            ),
        }
    )

    preds = triage_model.predict(prepared)
    probs = triage_model.predict_proba(prepared)

    urgency_list, risk_list, goodwill_list = [], [], []
    for i in range(len(prepared)):
        u_label = URGENCY_LABELS.get(int(preds[i]), str(preds[i]))
        r_pct, g_inr = compute_risk_and_goodwill(
            probs[i], prepared.loc[i, "price_inr"], prepared.loc[i, "days_since_delivery"]
        )
        urgency_list.append(u_label)
        risk_list.append(r_pct)
        goodwill_list.append(g_inr)

    out = df_in.copy()
    out["order_price_inr"] = prices_series.values
    out["predicted_urgency_category"] = urgency_list
    out["churn_risk_pct"] = risk_list
    out["suggested_goodwill_inr"] = goodwill_list
    out = out.sort_values(by=["churn_risk_pct", "suggested_goodwill_inr"], ascending=[False, False]).reset_index(
        drop=True
    )

    high_urg = int((out["predicted_urgency_category"] == "Negative / High Urgency").sum())
    at_risk_mask = out["churn_risk_pct"] >= 45.0
    at_risk = int(at_risk_mask.sum())
    low_risk = int((~at_risk_mask).sum())
    avg_churn_pct = round(float(out["churn_risk_pct"].mean()), 1) if len(out) > 0 else 0.0
    at_risk_revenue_inr = int(round(float(out.loc[at_risk_mask, "order_price_inr"].sum()))) if at_risk > 0 else 0
    total_goodwill_inr = int(out["suggested_goodwill_inr"].sum())

    return out, {
        "high_urgency": high_urg,
        "at_risk": at_risk,
        "low_risk": low_risk,
        "avg_churn_pct": avg_churn_pct,
        "at_risk_revenue_inr": at_risk_revenue_inr,
        "total_goodwill_inr": total_goodwill_inr,
    }


def render_prediction_graphs(scored_df):
    sns.set_theme(style="darkgrid")
    if len(scored_df) == 1:
        row = scored_df.iloc[0]
        fig, ax = plt.subplots(figsize=(6.8, 2.5))
        fig.patch.set_facecolor("#14102e")
        ax.set_facecolor("#1b153d")

        metrics = ["Churn Risk (%)", "Goodwill / 10 (₹)"]
        values = [float(row["churn_risk_pct"]), float(row["suggested_goodwill_inr"]) / 10.0]
        colors = ["#FF3366", "#00E5FF"]

        bars = ax.barh(metrics, values, color=colors, height=0.45)
        ax.set_xlim(0, max(100.0, max(values) * 1.18))
        ax.tick_params(colors="#f8fafc", labelsize=9)
        ax.set_title("Single Review Risk & Retention Visual", color="#ffffff", fontsize=11, fontweight="bold", pad=10)
        for bar, orig in zip(bars, [f"{row['churn_risk_pct']}%", f"₹{row['suggested_goodwill_inr']}"]):
            ax.text(
                bar.get_width() + 1.5,
                bar.get_y() + bar.get_height() / 2,
                orig,
                va="center",
                color="#ffffff",
                fontsize=9,
                fontweight="bold",
            )
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return

    c_left, c_right = st.columns(2)
    with c_left:
        fig1, ax1 = plt.subplots(figsize=(5.5, 3.2))
        fig1.patch.set_facecolor("#14102e")
        ax1.set_facecolor("#1b153d")

        counts_df = (
            scored_df["predicted_urgency_category"]
            .value_counts()
            .rename_axis("Urgency")
            .reset_index(name="Reviews")
        )
        sns.barplot(
            data=counts_df,
            x="Urgency",
            y="Reviews",
            hue="Urgency",
            palette=URGENCY_PALETTE,
            legend=False,
            ax=ax1,
        )
        ax1.tick_params(axis="x", colors="#f8fafc", labelsize=8, rotation=15)
        ax1.tick_params(axis="y", colors="#f8fafc", labelsize=8)
        ax1.set_xlabel("")
        ax1.set_ylabel("Count", color="#cbd5e1", fontsize=9)
        ax1.set_title("Reviews by Urgency Class", color="#ffffff", fontsize=10.5, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    with c_right:
        fig2, ax2 = plt.subplots(figsize=(5.5, 3.2))
        fig2.patch.set_facecolor("#14102e")
        ax2.set_facecolor("#1b153d")

        top_df = scored_df.head(8).copy()
        top_df["row_label"] = [f"#{i + 1}" for i in range(len(top_df))]
        sns.barplot(
            data=top_df,
            x="row_label",
            y="churn_risk_pct",
            hue="predicted_urgency_category",
            palette=URGENCY_PALETTE,
            dodge=False,
            ax=ax2,
        )
        ax2.tick_params(colors="#f8fafc", labelsize=8)
        ax2.set_xlabel("Top Priority Items", color="#cbd5e1", fontsize=9)
        ax2.set_ylabel("Churn Risk (%)", color="#cbd5e1", fontsize=9)
        ax2.set_title("Top Risk Scores (%)", color="#ffffff", fontsize=10.5, fontweight="bold")
        ax2.legend(title="", loc="upper right", fontsize=7.5, facecolor="#14102e", labelcolor="#ffffff")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)


triage_model = load_triage_pipeline()

b_col, n1, n2, n3, n4, badge_col = st.columns([2.2, 1, 1, 1.2, 1, 1.8])
with b_col:
    st.markdown('<div class="top-brand">✦ RetainCraft</div>', unsafe_allow_html=True)
with n1:
    if st.button("Home", type="primary" if st.session_state.nav_tab == "Home" else "secondary", use_container_width=True):
        st.session_state.nav_tab = "Home"
with n2:
    if st.button("About", type="primary" if st.session_state.nav_tab == "About" else "secondary", use_container_width=True):
        st.session_state.nav_tab = "About"
with n3:
    if st.button("How It Works", type="primary" if st.session_state.nav_tab == "How It Works" else "secondary", use_container_width=True):
        st.session_state.nav_tab = "How It Works"
with n4:
    if st.button("Insights", type="primary" if st.session_state.nav_tab == "Insights" else "secondary", use_container_width=True):
        st.session_state.nav_tab = "Insights"
with badge_col:
    st.markdown(
        '<div style="text-align:right; padding-top:8px; color:#cbd5e1; font-size:0.88rem;">✦ Built with ♥ using ML</div>',
        unsafe_allow_html=True,
    )

st.write("")

hero_left, hero_right = st.columns([1.3, 1], gap="large")
with hero_left:
    st.markdown(
        """
        <div class="main-hero-title">
            RetainCraft –<br>
            <span>Review Triage &amp; Retention Studio</span>
        </div>
        <div class="main-hero-sub">
            Turn customer reviews into actionable insights —<br>
            prioritize issues, reduce churn, and keep more customers.
        </div>
        """,
        unsafe_allow_html=True,
    )
    cta1, cta2, _ = st.columns([1.1, 1.1, 0.8])
    with cta1:
        if st.button("✦ Try the Model", type="primary", use_container_width=True):
            st.session_state.nav_tab = "Home"
    with cta2:
        if st.button("▶ Learn More", type="secondary", use_container_width=True):
            st.session_state.nav_tab = "How It Works"

with hero_right:
    st.markdown(
        """
        <div style="display:flex; justify-content:center; align-items:center; padding-top:4px;">
            <svg width="360" height="230" viewBox="0 0 360 230" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <radialGradient id="glow" cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stop-color="#8b5cf6" stop-opacity="0.55"/>
                        <stop offset="100%" stop-color="#090716" stop-opacity="0"/>
                    </radialGradient>
                    <linearGradient id="botBody" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#f8fafc"/>
                        <stop offset="100%" stop-color="#c4b5fd"/>
                    </linearGradient>
                    <linearGradient id="chartBar" x1="0%" y1="100%" x2="0%" y2="0%">
                        <stop offset="0%" stop-color="#4f46e5"/>
                        <stop offset="100%" stop-color="#38bdf8"/>
                    </linearGradient>
                </defs>
                <circle cx="195" cy="115" r="105" fill="url(#glow)"/>
                <rect x="65" y="18" width="110" height="36" rx="10" fill="#1e1b4b" stroke="#818cf8" stroke-width="1.5"/>
                <text x="120" y="41" text-anchor="middle" fill="#fbbf24" font-size="15" font-weight="bold">★ ★ ★ ★ ☆</text>
                <rect x="28" y="88" width="76" height="52" rx="14" fill="#fb7185"/>
                <polygon points="84,140 96,152 96,136" fill="#fb7185"/>
                <circle cx="56" cy="108" r="3.5" fill="#1e1b4b"/>
                <circle cx="76" cy="108" r="3.5" fill="#1e1b4b"/>
                <path d="M56 125 Q66 116 76 125" stroke="#1e1b4b" stroke-width="3" fill="none" stroke-linecap="round"/>
                <line x1="195" y1="30" x2="195" y2="48" stroke="#c4b5fd" stroke-width="4" stroke-linecap="round"/>
                <circle cx="195" cy="26" r="6" fill="#38bdf8"/>
                <rect x="145" y="48" width="100" height="74" rx="34" fill="url(#botBody)"/>
                <rect x="158" y="62" width="74" height="46" rx="22" fill="#0f172a"/>
                <path d="M172 86 Q179 76 186 86" stroke="#38bdf8" stroke-width="3.5" fill="none" stroke-linecap="round"/>
                <path d="M204 86 Q211 76 218 86" stroke="#38bdf8" stroke-width="3.5" fill="none" stroke-linecap="round"/>
                <rect x="155" y="128" width="80" height="62" rx="28" fill="url(#botBody)"/>
                <path d="M156 146 Q165 158 176 154" stroke="#a78bfa" stroke-width="6" fill="none" stroke-linecap="round"/>
                <path d="M234 146 Q225 158 214 154" stroke="#a78bfa" stroke-width="6" fill="none" stroke-linecap="round"/>
                <path d="M176 144 H182 L186 162 H207 L211 148 H184" stroke="#4f46e5" stroke-width="3" fill="#38bdf8" fill-opacity="0.25" stroke-linejoin="round" stroke-linecap="round"/>
                <circle cx="189" cy="168" r="3" fill="#1e1b4b"/>
                <circle cx="204" cy="168" r="3" fill="#1e1b4b"/>
                <rect x="268" y="138" width="16" height="46" rx="4" fill="url(#chartBar)"/>
                <rect x="292" y="118" width="16" height="66" rx="4" fill="url(#chartBar)"/>
                <rect x="316" y="94" width="16" height="90" rx="4" fill="url(#chartBar)"/>
                <path d="M262 126 Q296 108 332 72" stroke="#38bdf8" stroke-width="4" fill="none" stroke-linecap="round"/>
                <polygon points="336,66 322,70 332,80" fill="#38bdf8"/>
            </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )

if st.session_state.nav_tab == "About":
    st.info(
        "**About RetainCraft:** Helps support and retention teams spot urgent complaints early, estimate customer churn risk, and size fair INR goodwill credits consistently."
    )
elif st.session_state.nav_tab == "How It Works":
    st.info(
        "**How It Works:** (1) Paste customer reviews line by line or upload a CSV file. (2) Your trained scikit-learn triage pipeline predicts urgency classes. (3) RetainCraft combines urgency probabilities with delivery delay and order price to rank high-risk accounts."
    )
elif st.session_state.nav_tab == "Insights" and st.session_state.scored_table is not None:
    render_prediction_graphs(st.session_state.scored_table)
    st.dataframe(st.session_state.scored_table.head(25), use_container_width=True)

with st.container(border=True):
    left_box, right_box = st.columns([1.15, 1], gap="large")

    with left_box:
        st.markdown('<div class="card-title">✦ Paste Your Reviews</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Add customer reviews or upload a CSV file</div>', unsafe_allow_html=True)

        pasted_text = st.text_area(
            "Paste reviews (one per line)",
            value=st.session_state.pasted_reviews,
            placeholder="Enter reviews here... (one per line)",
            height=115,
            label_visibility="collapsed",
        )
        st.session_state.pasted_reviews = pasted_text

        st.markdown('<div class="or-divider">OR</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title" style="font-size:1rem;">☁ Upload CSV File</div>', unsafe_allow_html=True)
        uploaded_csv = st.file_uploader(
            "Drag and drop or click to upload",
            type=["csv"],
            label_visibility="collapsed",
        )

    with right_box:
        analyze_clicked = st.button("✦ Analyze Reviews", type="primary", use_container_width=True)

        st.markdown('<div class="card-title" style="margin-top:14px;">📊 Results</div>', unsafe_allow_html=True)

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

        active_summary = {
            "high_urgency": high_val,
            "at_risk": at_risk_val,
            "avg_churn_pct": avg_churn_val,
            "at_risk_revenue_inr": at_risk_rev_val,
            "total_goodwill_inr": total_goodwill_val,
        }

        def fmt_int(val):
            return f"{int(val):,}" if isinstance(val, (int, float)) and not isinstance(val, bool) else str(val)

        def fmt_inr(val):
            return f"₹{int(round(val)):,}" if isinstance(val, (int, float)) and not isinstance(val, bool) else str(val)

        def fmt_pct(val):
            return f"{float(val):.1f}%" if isinstance(val, (int, float)) and not isinstance(val, bool) else str(val)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                f"""
                <div class="res-high">
                    <div class="res-label">⚠ High Urgency</div>
                    <div class="res-num">{fmt_int(high_val)}</div>
                    <div class="res-caption">Needs Attention</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f"""
                <div class="res-mid">
                    <div class="res-label">📈 At Risk</div>
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

        st.write("")

        v1, v2 = st.columns(2)
        with v1:
            st.markdown(
                f"""
                <div class="val-revenue">
                    <div class="res-label">₹ At-Risk Revenue Value</div>
                    <div class="res-num">₹{at_risk_rev_val:,}</div>
                    <div class="res-caption">Order value across elevated-risk reviews</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with v2:
            st.markdown(
                f"""
                <div class="val-churn">
                    <div class="res-label">📉 Average Churn Risk</div>
                    <div class="res-num">{avg_churn_val}%</div>
                    <div class="res-caption">Mean churn likelihood in current batch</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        active_summary = summary if summary else {
            "high_urgency": high_val,
            "at_risk": at_risk_val,
            "avg_churn_pct": avg_churn_val,
            "at_risk_revenue_inr": at_risk_rev_val,
            "total_goodwill_inr": 84200,
        }
        suggestion_items = "".join(f"<li>{item}</li>" for item in build_suggestions(active_summary))
        st.markdown(
            f"""
            <div class="suggestions-box">
                <div class="suggestions-title">💡 Retention &amp; Action Suggestions</div>
                <ul class="suggestions-list">{suggestion_items}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.scored_table is not None:
            st.write("")
            render_prediction_graphs(st.session_state.scored_table)
            st.dataframe(st.session_state.scored_table.head(10), use_container_width=True)
            st.download_button(
                "Download Scored CSV",
                data=st.session_state.scored_table.to_csv(index=False).encode("utf-8"),
                file_name="retaincraft_scored_reviews.csv",
                mime="text/csv",
            )

st.write("")

f1, f2, f3, f4 = st.columns(4)
with f1:
    st.markdown('<div class="footer-pill">✦ Why RetainCraft?</div>', unsafe_allow_html=True)
with f2:
    st.markdown('<div class="footer-pill">⚡ Smart Predictions</div>', unsafe_allow_html=True)
with f3:
    st.markdown('<div class="footer-pill">🎯 Better Retention</div>', unsafe_allow_html=True)
with f4:
    st.markdown('<div class="footer-pill">😊 Happier Customers</div>', unsafe_allow_html=True)