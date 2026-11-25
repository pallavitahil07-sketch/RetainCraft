import json
import os
from pathlib import Path
import time
from google import genai
from google.genai import types
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
GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-2.5-flash"]


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
        try:
            return joblib.load(model_path)
        except Exception:
            return None
    return None


def _find_best_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for col in df.columns:
        col_l = str(col).strip().lower()
        if any(cand.lower() in col_l for cand in candidates):
            return col
    return None


def score_reviews_dataframe(df_in: pd.DataFrame):
    df_work = df_in.copy()
    n_rows = len(df_work)

    if n_rows == 0:
        empty_summary = {
            "high_urgency": 0,
            "at_risk": 0,
            "low_risk": 0,
            "at_risk_revenue_inr": 0.0,
            "avg_churn_pct": 0.0,
            "total_goodwill_inr": 0.0,
        }
        return df_work, empty_summary

    text_col = _find_best_column(
        df_work,
        [
            "review_text", "review", "feedback", "comment", "comments",
            "text", "body", "message", "description", "summary", "title", "notes",
        ],
    )
    if text_col is not None:
        review_series = df_work[text_col].fillna("").astype(str)
    else:
        obj_cols = df_work.select_dtypes(include=["object", "string"]).columns.tolist()
        if obj_cols:
            review_series = (
                df_work[obj_cols].fillna("").astype(str).agg(" - ".join, axis=1).str.strip(" -")
            )
        else:
            review_series = pd.Series([f"Row {i + 1}" for i in range(n_rows)], index=df_work.index)

    df_work["review_text"] = review_series

    if "review_id" not in df_work.columns:
        df_work["review_id"] = [f"REV-{i + 1:04d}" for i in range(n_rows)]

    price_col = _find_best_column(
        df_work,
        ["price_inr", "order_value", "amount", "price", "revenue", "total", "cost", "fare", "value"],
    )
    if price_col is not None:
        prices_series = pd.to_numeric(df_work[price_col], errors="coerce").fillna(DEFAULT_PRICE)
    else:
        num_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            prices_series = pd.to_numeric(df_work[num_cols[0]], errors="coerce").abs().fillna(DEFAULT_PRICE)
        else:
            prices_series = pd.Series(DEFAULT_PRICE, index=df_work.index, dtype="float64")

    days_col = _find_best_column(
        df_work,
        ["days_since_delivery", "delivery_days", "days", "delay", "tat", "age", "tenure"],
    )
    if days_col is not None:
        days_series = pd.to_numeric(df_work[days_col], errors="coerce").fillna(DEFAULT_DAYS)
    else:
        days_series = pd.Series(DEFAULT_DAYS, index=df_work.index, dtype="float64")

    cat_col = _find_best_column(
        df_work, ["category", "product_category", "segment", "type", "department", "plan"]
    )
    category_series = (
        df_work[cat_col].fillna("General").astype(str)
        if cat_col is not None
        else pd.Series("General", index=df_work.index)
    )

    city_col = _find_best_column(df_work, ["city", "location", "region", "state", "country"])
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
    urgency_labels = []
    churn_risk_pct = None

    if model is not None:
        try:
            preds = model.predict(df_work)
            for p in preds:
                p_str = str(p).strip().lower()
                if "high" in p_str or p_str in {"2", "urgent", "critical"}:
                    urgency_labels.append("High Urgency")
                elif "risk" in p_str or "med" in p_str or p_str in {"1", "at_risk", "at risk"}:
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
                if len(probs) == n_rows:
                    max_probs = np.max(probs, axis=1)
                    churn_risk_pct = np.clip(np.round(max_probs * 100, 1), 1.0, 99.0)
            except Exception:
                churn_risk_pct = None

    if not urgency_labels:
        urgent_words = (
            "broken", "damaged", "late", "delay", "refund", "fake", "counterfeit",
            "heat", "burnt", "worst", "missing", "leaking", "horrible", "refused", "error", "fail",
        )
        risk_words = (
            "dull", "tight", "average", "flimsy", "missing", "delay", "okay", "dented", "slow", "issue",
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
        p_val = float(price) if pd.notna(price) else DEFAULT_PRICE
        if lbl == "High Urgency":
            goodwill_inr.append(round(min(max(p_val * 0.20, 150.0), 1500.0)))
        elif lbl == "At Risk":
            goodwill_inr.append(round(min(max(p_val * 0.10, 75.0), 750.0)))
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


def _get_raw_gemini_key() -> str:
    val = ""
    try:
        raw = st.secrets.get("GEMINI_API_KEY", "")
        if raw is not None:
            val = str(raw).strip()
    except Exception:
        val = ""
    if not val:
        val = str(os.getenv("GEMINI_API_KEY", "")).strip()
    return val


def get_gemini_client():
    api_key = _get_raw_gemini_key()
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def call_gemini_with_fallback(contents, config=None) -> str | None:
    client = get_gemini_client()
    if client is None:
        return None

    for model_name in GEMINI_MODELS:
        for attempt in range(2):
            try:
                if config is not None:
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                else:
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                    )
                txt = (resp.text or "").strip()
                if txt:
                    return txt
            except Exception as e:
                err_str = str(e)
                is_transient = ("503" in err_str) or ("UNAVAILABLE" in err_str) or ("429" in err_str)
                if is_transient and attempt == 0:
                    time.sleep(0.8)
                    continue
                break
    return None


def _fallback_theme(text: str) -> str:
    t = str(text).lower()
    if any(k in t for k in ["late", "delay", "slow", "tracking", "delivery"]):
        return "Logistics & Delivery Delay"
    if any(k in t for k in ["broken", "damaged", "leaking", "dented", "burnt", "heat"]):
        return "Damaged / Defective Item"
    if any(k in t for k in ["fake", "counterfeit", "wrong item", "missing"]):
        return "Authenticity / Fulfillment Mismatch"
    if any(k in t for k in ["tight", "size", "fit", "stiff", "flimsy"]):
        return "Product Fit & Build Quality"
    if any(k in t for k in ["refund", "support", "refused", "reply"]):
        return "Support & Refund Friction"
    return "Positive / General Feedback"


def tag_root_causes_and_chart(scored_df: pd.DataFrame, use_gemini: bool = True) -> pd.DataFrame:
    df_out = scored_df.copy()
    texts = df_out["review_text"].fillna("").astype(str).tolist()
    themes = [_fallback_theme(t) for t in texts]

    if use_gemini and len(texts) > 0 and get_gemini_client() is not None:
        sample_items = [
            {"idx": i, "text": texts[i][:180], "urgency": str(df_out.iloc[i]["urgency_class"])}
            for i in range(min(len(texts), 30))
        ]
        allowed_themes = [
            "Logistics & Delivery Delay",
            "Damaged / Defective Item",
            "Authenticity / Fulfillment Mismatch",
            "Product Fit & Build Quality",
            "Support & Refund Friction",
            "Positive / General Feedback",
        ]
        prompt = (
            f"Classify each item into one of these exact themes: {allowed_themes}. "
            f'Return ONLY a valid JSON list of objects with keys "idx" (int) and "theme" (str). '
            f"Items: {json.dumps(sample_items)}"
        )
        raw_json = call_gemini_with_fallback(
            prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        if raw_json:
            try:
                parsed = json.loads(raw_json)
                if isinstance(parsed, list):
                    for item in parsed:
                        idx = item.get("idx")
                        th = item.get("theme")
                        if isinstance(idx, int) and 0 <= idx < len(themes) and th in allowed_themes:
                            themes[idx] = th
            except Exception:
                pass

    df_out["root_cause_theme"] = themes
    return df_out


def _local_insights_fallback(scored_df: pd.DataFrame, summary: dict, file_label: str) -> str:
    top_theme = "General Feedback"
    if "root_cause_theme" in scored_df.columns and not scored_df["root_cause_theme"].empty:
        top_theme = str(scored_df["root_cause_theme"].value_counts().index[0])

    top_cat = "General"
    if "category" in scored_df.columns and not scored_df["category"].empty:
        top_cat = str(scored_df["category"].value_counts().index[0])

    return (
        f"- **Batch Overview ({file_label}):** Scored **{fmt_int(len(scored_df))}** rows — "
        f"**{fmt_int(summary.get('high_urgency'))}** High Urgency, **{fmt_int(summary.get('at_risk'))}** At Risk, "
        f"and **{fmt_int(summary.get('low_risk'))}** Low Risk (lead theme: **{top_theme}**, top segment: **{top_cat}**).\n"
        f"- **Revenue & Churn Exposure:** **{fmt_inr(summary.get('at_risk_revenue_inr'))}** in elevated-risk order/ticket value "
        f"with an average churn risk of **{fmt_pct(summary.get('avg_churn_pct'))}**.\n"
        f"- **Immediate Priority:** Resolve High-Urgency complaints first and follow up on At-Risk accounts "
        f"(estimated goodwill pool: **{fmt_inr(summary.get('total_goodwill_inr'))}**).\n"
        f"- **Next Step:** Filter high-churn rows below, draft responses for top tickets, or click **✦ Analyze Reviews** again once Gemini capacity clears."
    )


def generate_genai_insights(scored_df: pd.DataFrame, summary: dict, file_label: str = "uploaded dataset") -> str:
    numeric_cols = scored_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_stats = {}
    for col in numeric_cols[:8]:
        s = pd.to_numeric(scored_df[col], errors="coerce").dropna()
        if not s.empty:
            numeric_stats[col] = {
                "mean": round(float(s.mean()), 2),
                "min": round(float(s.min()), 2),
                "max": round(float(s.max()), 2),
                "sum": round(float(s.sum()), 2),
            }

    sample_rows = scored_df.head(5).to_dict(orient="records")
    prompt = f"""
You are a retention and business analytics advisor inside RetainCraft.
Analyze the dataset summary below and return clear, practical business insights and retention recommendations.
Adapt naturally to whatever columns and domain exist in this dataset.

Dataset Source: {file_label}
Total Rows Scored: {len(scored_df)}
Columns: {list(scored_df.columns)}
Triage Summary: {summary}
Numeric Column Stats: {numeric_stats}
Sample Rows (up to 5): {sample_rows}

Provide 4 concise bullet points:
1. Key patterns or standout findings in this data.
2. Main risk areas (churn, delays, complaints, or revenue exposure).
3. Immediate priority retention & customer support actions.
4. Practical next steps to improve retention and outcomes.
""".strip()

    ai_text = call_gemini_with_fallback(prompt)
    return ai_text if ai_text else _local_insights_fallback(scored_df, summary, file_label)


def generate_retention_playbook(scored_df: pd.DataFrame, summary: dict) -> str:
    sample_rows = scored_df.head(6).to_dict(orient="records")
    prompt = f"""
Create a concise **Retention Playbook & Offer Strategy** for these triage results.
Recommend when to use non-monetary retention actions (priority replacement, guided onboarding, loyalty points) vs. goodwill credits so margins stay protected.

Summary: {summary}
Sample Scored Rows: {sample_rows}
""".strip()

    ai_text = call_gemini_with_fallback(prompt)
    if ai_text:
        return ai_text

    return (
        f"- **High Urgency ({fmt_int(summary.get('high_urgency'))}):** Trigger immediate replacement/refund or priority callback; "
        f"cap goodwill credits on high-ticket items to protect margin.\n"
        f"- **At Risk ({fmt_int(summary.get('at_risk'))}):** Send proactive apology, delivery/fit help, or targeted loyalty/voucher offers "
        f"across **{fmt_inr(summary.get('at_risk_revenue_inr'))}** in elevated-risk value.\n"
        f"- **Low Risk ({fmt_int(summary.get('low_risk'))}):** Use low-cost engagement or repeat-purchase nudges rather than discounting."
    )


def draft_customer_and_escalation(row_dict: dict) -> str:
    prompt = f"""
You are a customer retention specialist for an e-commerce & marketplace operations team.
Using this record, write two short sections:
1. **Customer Reply** (empathetic, clear, action-oriented message to the customer)
2. **Internal Escalation Note** (concise ops brief with priority, root issue, and goodwill guidance)

Record Details: {row_dict}
""".strip()

    ai_text = call_gemini_with_fallback(prompt)
    if ai_text:
        return ai_text

    return (
        "**Customer Reply Draft:** We sincerely apologize for the issue with your recent order "
        f"({row_dict.get('review_id', 'Ticket')}). Our priority care team is reviewing this now and will follow up promptly.\n\n"
        "**Internal Escalation Brief:** Urgency="
        f"{row_dict.get('urgency_class', 'N/A')}, Churn Risk={row_dict.get('churn_risk_pct', 'N/A')}%, "
        f"Suggested Goodwill={fmt_inr(row_dict.get('goodwill_credit_inr', 0))}."
    )


def ask_data_copilot(question: str, scored_df: pd.DataFrame, summary: dict) -> str:
    theme_counts = (
        scored_df["root_cause_theme"].value_counts().to_dict()
        if "root_cause_theme" in scored_df.columns
        else {}
    )
    cat_counts = (
        scored_df["category"].value_counts().head(6).to_dict()
        if "category" in scored_df.columns
        else {}
    )
    sample_rows = scored_df.head(8).to_dict(orient="records")

    prompt = f"""
Answer the user's analytics/retention question based on the scored dataset below. Be concise, accurate, and actionable.

Question: {question}
Total Rows: {len(scored_df)}
Summary Metrics: {summary}
Root-Cause Themes: {theme_counts}
Top Categories: {cat_counts}
Sample Rows: {sample_rows}
""".strip()

    ai_text = call_gemini_with_fallback(prompt)
    if ai_text:
        return ai_text

    top_themes_str = ", ".join(f"{k} ({v})" for k, v in list(theme_counts.items())[:3]) or "N/A"
    return (
        f"Gemini is temporarily busy right now, so here is a quick summary for **{question}**:\n"
        f"- **Rows Scored:** {fmt_int(len(scored_df))} ({fmt_int(summary.get('high_urgency'))} High Urgency, "
        f"{fmt_int(summary.get('at_risk'))} At Risk, {fmt_int(summary.get('low_risk'))} Low Risk)\n"
        f"- **Elevated Revenue Exposure:** {fmt_inr(summary.get('at_risk_revenue_inr'))} | **Avg Churn Risk:** {fmt_pct(summary.get('avg_churn_pct'))}\n"
        f"- **Top Themes:** {top_themes_str}"
    )


def inspect_defect_photo_with_gemini(image_bytes: bytes, mime_type: str, context_note: str) -> str:
    prompt = f"""
Inspect this uploaded product/packaging photo alongside the support context below.
Provide:
1. Visible condition / damage / packaging observations.
2. Severity rating (Low / Medium / High) and churn risk impact.
3. Recommended resolution (Instant Replacement, Partial Refund, Goodwill Credit, or No Action).

Context: {context_note or "Customer defect/review inspection"}
""".strip()

    ai_text = call_gemini_with_fallback(
        [
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ]
    )
    if ai_text:
        return ai_text

    return (
        "Gemini vision is temporarily experiencing high demand. Please retry **Inspect Defect Photo** in a moment, "
        f"or log context (`{context_note or 'No note provided'}`) alongside the photo for manual QA review."
    )


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

for k, v in {
    "scored_table": None,
    "scored_summary": None,
    "genai_insights": None,
    "playbook_text": None,
    "reply_draft": None,
    "copilot_answer": None,
    "defect_report": None,
    "uploaded_df": None,
    "uploaded_name": None,
    "active_tab": "Home",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

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
    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input {
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
    st.markdown('<div class="right-badge">✦ Built with ♥ using ML & Gen AI</div>', unsafe_allow_html=True)

st.markdown("---")

if st.session_state.active_tab == "About":
    st.subheader("About RetainCraft")
    st.write(
        "RetainCraft pairs your trained ML triage model with Gemini Gen AI to help support and "
        "operations teams prioritize tickets, inspect defect media, tag root causes, and retain high-value customers."
    )
elif st.session_state.active_tab == "How It Works":
    st.subheader("How It Works")
    st.write(
        "1. Paste reviews or upload any CSV file.\n"
        "2. Click **✦ Analyze Reviews** to score urgency, churn risk, root-cause themes, and Gen AI insights.\n"
        "3. Use the Reply Drafter, Retention Playbook, Ask Your Data Copilot, or Defect Photo Inspector below."
    )
elif st.session_state.active_tab == "Insights":
    st.subheader("Scored Queue & AI Business Insights")
    if st.session_state.scored_table is not None and not st.session_state.scored_table.empty:
        if st.session_state.get("genai_insights"):
            st.markdown("### ✦ Gen AI Business Insights")
            st.markdown(st.session_state.genai_insights)
        if "root_cause_theme" in st.session_state.scored_table.columns:
            st.markdown("### Root-Cause Theme Breakdown")
            st.bar_chart(st.session_state.scored_table["root_cause_theme"].value_counts())
        st.markdown("### Scored Data Table")
        st.dataframe(st.session_state.scored_table, width="stretch")
        csv_bytes = st.session_state.scored_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Scored CSV",
            data=csv_bytes,
            file_name="retaincraft_scored_output.csv",
            mime="text/csv",
        )
    else:
        st.info("No dataset analyzed yet. Go to Home, paste reviews or upload any CSV, and click Analyze Reviews.")
else:
    hero_left, hero_right = st.columns([1.4, 1.0], gap="large")
    with hero_left:
        st.markdown('<div class="hero-title">RetainCraft –<br>Review Triage & Retention Studio</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-sub">Turn customer reviews & CSV datasets into actionable insights —<br>prioritize issues, reduce churn, and keep more customers.</div>',
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
        st.caption("Add customer reviews or upload any CSV file")
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
                scored_df = tag_root_causes_and_chart(scored_df, use_gemini=True)
                batch_label = st.session_state.get("uploaded_name") or "Uploaded CSV"
            else:
                lines = [line.strip() for line in pasted_text.splitlines() if line.strip()]
                if lines:
                    df_input = pd.DataFrame({"review_text": lines})
                    scored_df, summary = score_reviews_dataframe(df_input)
                    scored_df = tag_root_causes_and_chart(scored_df, use_gemini=True)
                    batch_label = "Pasted Reviews"
                else:
                    scored_df, summary, batch_label = None, None, None
                    st.warning("Paste at least one review line or upload a CSV file to see predictions.")

            if scored_df is not None and summary is not None:
                st.session_state.scored_table = scored_df
                st.session_state.scored_summary = summary
                st.session_state.genai_insights = generate_genai_insights(scored_df, summary, batch_label)
                st.session_state.playbook_text = generate_retention_playbook(scored_df, summary)
                st.session_state.reply_draft = None
                st.session_state.copilot_answer = None
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
                    <div class="res-caption">Order value across elevated-risk rows</div>
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
        st.markdown("**✦ Retention & Action Suggestions (ML + Gen AI)**")
        if st.session_state.get("genai_insights"):
            st.markdown(st.session_state.genai_insights)
        elif st.session_state.get("scored_summary"):
            st.write(
                f"- **Escalate {fmt_int(high_val)} High-Urgency ticket(s) first:** Route critical items to priority support promptly.\n"
                f"- **Protect {fmt_inr(at_risk_rev_val)} in At-Risk value:** Follow up with {fmt_int(at_risk_val)} elevated-risk rows "
                f"(avg churn risk {fmt_pct(avg_churn_val)}; goodwill pool {fmt_inr(total_goodwill_inr)})."
            )
        else:
            st.caption("No retention actions generated yet. Paste reviews or upload a CSV and click Analyze Reviews.")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.scored_table is not None and not st.session_state.scored_table.empty:
            df_scored = st.session_state.scored_table

            if "root_cause_theme" in df_scored.columns:
                st.markdown("#### ✦ Root-Cause & Theme Breakdown")
                st.bar_chart(df_scored["root_cause_theme"].value_counts())

            if st.session_state.get("playbook_text"):
                st.markdown('<div class="sugg-box">', unsafe_allow_html=True)
                st.markdown("**✦ Retention Playbook & Offer Optimizer**")
                st.markdown(st.session_state.playbook_text)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("#### ✦ One-Click Customer Reply & Escalation Drafter")
            ticket_options = [
                f"{row['review_id']} | {row['urgency_class']} | {str(row['review_text'])[:45]}"
                for _, row in df_scored.head(50).iterrows()
            ]
            selected_ticket = st.selectbox("Select a ticket/review to draft response", ticket_options)
            if st.button("Draft Customer Reply & Escalation", key="btn_draft_reply"):
                sel_idx = ticket_options.index(selected_ticket)
                st.session_state.reply_draft = draft_customer_and_escalation(df_scored.iloc[sel_idx].to_dict())
            if st.session_state.get("reply_draft"):
                st.markdown('<div class="sugg-box">', unsafe_allow_html=True)
                st.markdown(st.session_state.reply_draft)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("#### ✦ Ask Your Data (Retention Copilot)")
            copilot_q = st.text_input(
                "Ask a question about this batch",
                placeholder="e.g., What is the biggest complaint theme and how should we prioritize retention?",
            )
            if st.button("Ask Copilot", key="btn_ask_copilot"):
                if copilot_q.strip():
                    st.session_state.copilot_answer = ask_data_copilot(
                        copilot_q.strip(), df_scored, st.session_state.scored_summary or {}
                    )
                else:
                    st.warning("Type a question first.")
            if st.session_state.get("copilot_answer"):
                st.markdown('<div class="sugg-box">', unsafe_allow_html=True)
                st.markdown(st.session_state.copilot_answer)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("#### ✦ Defect & Packaging Photo Inspector (Multimodal Gen AI)")
            defect_photo = st.file_uploader(
                "Upload product defect or packaging photo (optional)",
                type=["png", "jpg", "jpeg", "webp"],
                key="defect_photo_uploader",
            )
            defect_note = st.text_input(
                "Optional defect context",
                placeholder="e.g., Customer reported item arrived damaged after 6 days",
            )
            if st.button("Inspect Defect Photo", key="btn_inspect_photo"):
                if defect_photo is not None:
                    mime = defect_photo.type or "image/jpeg"
                    st.session_state.defect_report = inspect_defect_photo_with_gemini(
                        defect_photo.getvalue(), mime, defect_note
                    )
                else:
                    st.warning("Upload a PNG, JPG, or WEBP image first.")
            if st.session_state.get("defect_report"):
                st.markdown('<div class="sugg-box">', unsafe_allow_html=True)
                st.markdown(st.session_state.defect_report)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("#### Scored Review Analysis")
            st.dataframe(df_scored.head(25), width="stretch")
            csv_bytes = df_scored.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Full Scored CSV",
                data=csv_bytes,
                file_name="retaincraft_scored_output.csv",
                mime="text/csv",
            )

        st.markdown("</div>", unsafe_allow_html=True)