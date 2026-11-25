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
import streamlit.components.v1 as components

st.set_page_config(
    page_title="RetainCraft",
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


def build_category_scorecards(scored_df: pd.DataFrame) -> list[dict]:
    theme_specs = [
        (
            "Logistics & Delivery Delay",
            "Delivery speed, tracking updates, and dispatch timing.",
            "Timely arrivals keep repeat orders steady; delays raise churn.",
        ),
        (
            "Damaged / Defective Item",
            "Item condition, safety, and outer packaging integrity.",
            "Damaged or faulty goods need quick replacement to save trust.",
        ),
        (
            "Authenticity / Fulfillment Mismatch",
            "Exact item match, completeness, and genuine quality.",
            "Wrong or missing items require fast resolution and verification.",
        ),
        (
            "Product Fit & Build Quality",
            "Sizing accuracy, sturdiness, and material feel.",
            "Fit or finish issues can often be solved with exchange or guidance.",
        ),
        (
            "Support & Refund Friction",
            "Resolution speed, refund clarity, and support responsiveness.",
            "Smooth support turns complaints into loyal repeat customers.",
        ),
        (
            "Positive / General Feedback",
            "Overall satisfaction, smooth delivery, and repeat intent.",
            "Healthy baseline feedback to reinforce with loyalty nudges.",
        ),
    ]
    total = max(len(scored_df), 1)
    counts = (
        scored_df["root_cause_theme"].value_counts().to_dict()
        if "root_cause_theme" in scored_df.columns
        else {}
    )
    cards = []
    for name, focus, note in theme_specs:
        cnt = int(counts.get(name, 0))
        share = cnt / total
        if name == "Positive / General Feedback":
            score_5 = round(min(5.0, max(2.5, 3.2 + share * 3.0)), 1)
        else:
            score_5 = round(max(1.5, min(4.8, 4.7 - share * 4.5)), 1)
        cards.append(
            {
                "theme": name,
                "count": cnt,
                "score": f"{score_5}/5",
                "summary": f"{cnt} mentions ({share * 100:.1f}%) — {focus} {note}",
            }
        )
    return cards


def build_strengths_and_complaints(scored_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    low_df = scored_df[scored_df["urgency_class"] == "Low Risk"]
    risk_df = scored_df[scored_df["urgency_class"].isin(["High Urgency", "At Risk"])]

    strengths = []
    for txt in low_df["review_text"].dropna().astype(str).head(4):
        clean = txt.strip()
        if clean:
            strengths.append(clean[:110])
    if not strengths:
        strengths = ["Consistent fulfillment across low-risk rows."]

    complaints = []
    if "root_cause_theme" in risk_df.columns and not risk_df.empty:
        for th, cnt in risk_df["root_cause_theme"].value_counts().head(4).items():
            if th != "Positive / General Feedback":
                complaints.append(f"{th} ({int(cnt)})")
    if not complaints:
        for txt in risk_df["review_text"].dropna().astype(str).head(4):
            clean = txt.strip()
            if clean:
                complaints.append(clean[:110])
    if not complaints:
        complaints = ["No elevated-risk complaints in this batch."]

    return strengths, complaints


def _local_insights_fallback(scored_df: pd.DataFrame, summary: dict, file_label: str) -> str:
    top_theme = "General Feedback"
    if "root_cause_theme" in scored_df.columns and not scored_df["root_cause_theme"].empty:
        top_theme = str(scored_df["root_cause_theme"].value_counts().index[0])

    return (
        f"- **Batch ({file_label}):** {fmt_int(len(scored_df))} rows — "
        f"{fmt_int(summary.get('high_urgency'))} High Urgency, {fmt_int(summary.get('at_risk'))} At Risk, "
        f"{fmt_int(summary.get('low_risk'))} Low Risk (top theme: {top_theme}).\n"
        f"- **Exposure:** {fmt_inr(summary.get('at_risk_revenue_inr'))} at-risk value; "
        f"{fmt_pct(summary.get('avg_churn_pct'))} avg churn risk; "
        f"{fmt_inr(summary.get('total_goodwill_inr'))} goodwill pool."
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
Analyze this customer review & retention batch and return 4 concise bullet points:
1. Main patterns across customer feedback and metrics.
2. Key complaint and churn risk drivers.
3. Priority support & retention actions.
4. Practical next steps for the team.

Source: {file_label}
Rows: {len(scored_df)}
Summary: {summary}
Numeric Stats: {numeric_stats}
Sample Rows: {sample_rows}
""".strip()

    ai_text = call_gemini_with_fallback(prompt)
    return ai_text if ai_text else _local_insights_fallback(scored_df, summary, file_label)


def generate_retention_playbook(scored_df: pd.DataFrame, summary: dict) -> str:
    sample_rows = scored_df.head(6).to_dict(orient="records")
    prompt = f"""
Create a concise Retention Playbook & Offer Strategy for these triage results.
Show when to use non-monetary retention actions vs. goodwill credits to protect margin.

Summary: {summary}
Sample Rows: {sample_rows}
""".strip()

    ai_text = call_gemini_with_fallback(prompt)
    if ai_text:
        return ai_text

    return (
        f"- **High Urgency ({fmt_int(summary.get('high_urgency'))}):** Fast replacement/refund + priority follow-up with capped goodwill.\n"
        f"- **At Risk ({fmt_int(summary.get('at_risk'))}):** Proactive outreach, delivery/fit fix, and targeted loyalty offer.\n"
        f"- **Low Risk ({fmt_int(summary.get('low_risk'))}):** Low-cost repeat-purchase or loyalty nudge."
    )


def draft_customer_and_escalation(row_dict: dict) -> str:
    prompt = f"""
Write two short sections for this ticket:
1. **Customer Reply** (empathetic, clear, action-oriented)
2. **Internal Escalation Note** (priority, root cause, goodwill guidance)

Record: {row_dict}
""".strip()

    ai_text = call_gemini_with_fallback(prompt)
    if ai_text:
        return ai_text

    return (
        "**Customer Reply:** We apologize for the issue with your order "
        f"({row_dict.get('review_id', 'Ticket')}) and are prioritizing a resolution now.\n\n"
        "**Internal Escalation Note:** Urgency="
        f"{row_dict.get('urgency_class', 'N/A')}, Churn Risk={row_dict.get('churn_risk_pct', 'N/A')}%, "
        f"Goodwill={fmt_inr(row_dict.get('goodwill_credit_inr', 0))}."
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
Answer this question concisely and practically based on the scored dataset:

Question: {question}
Total Rows: {len(scored_df)}
Summary: {summary}
Themes: {theme_counts}
Categories: {cat_counts}
Sample Rows: {sample_rows}
""".strip()

    ai_text = call_gemini_with_fallback(prompt)
    if ai_text:
        return ai_text

    top_themes_str = ", ".join(f"{k} ({v})" for k, v in list(theme_counts.items())[:3]) or "N/A"
    return (
        f"**Summary for:** {question}\n"
        f"- **Rows:** {fmt_int(len(scored_df))} ({fmt_int(summary.get('high_urgency'))} High Urgency, "
        f"{fmt_int(summary.get('at_risk'))} At Risk, {fmt_int(summary.get('low_risk'))} Low Risk)\n"
        f"- **At-Risk Value:** {fmt_inr(summary.get('at_risk_revenue_inr'))} | **Top Themes:** {top_themes_str}"
    )


def inspect_defect_photo_with_gemini(image_bytes: bytes, mime_type: str, context_note: str) -> str:
    prompt = f"""
Inspect this product/packaging photo and note:
1. Visible condition or damage
2. Severity (Low / Medium / High) & churn impact
3. Recommended resolution

Context: {context_note or "Defect/packaging check"}
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
        "Unable to reach Gemini vision right now. Please try **Inspect Photo** again shortly "
        f"({context_note or 'no extra note'})."
    )


def render_robot_basket_widget(boost: bool = False):
    boost_js = "true" if boost else "false"
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
          background: transparent;
          display: flex;
          align-items: center;
          justify-content: center;
          height: 220px;
          overflow: hidden;
          font-family: system-ui, -apple-system, sans-serif;
        }}
        .stage {{
          position: relative;
          width: 100%;
          max-width: 420px;
          height: 210px;
          border-radius: 20px;
          background: radial-gradient(circle at 60% 45%, rgba(164, 108, 66, 0.16), rgba(251, 248, 244, 0));
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
        }}
        @keyframes glide {{
          0%, 100% {{ transform: translateX(0px) translateY(0px); }}
          50% {{ transform: translateX(10px) translateY(-3px); }}
        }}
        @keyframes wheelSpin {{
          0% {{ transform: rotate(0deg); }}
          100% {{ transform: rotate(360deg); }}
        }}
        @keyframes floatStar {{
          0%, 100% {{ transform: translateY(0px) scale(1); opacity: 0.9; }}
          50% {{ transform: translateY(-6px) scale(1.15); opacity: 1; }}
        }}
        @keyframes burstMove {{
          0% {{ transform: translateX(0px) rotate(0deg) scale(1); }}
          25% {{ transform: translateX(18px) rotate(3deg) scale(1.05); }}
          50% {{ transform: translateX(-10px) rotate(-3deg) scale(1.02); }}
          75% {{ transform: translateX(14px) rotate(2deg) scale(1.04); }}
          100% {{ transform: translateX(0px) rotate(0deg) scale(1); }}
        }}
        .mover {{
          animation: glide 2.5s ease-in-out infinite;
          transform-origin: center;
        }}
        .stage.burst .mover {{
          animation: burstMove 0.85s ease-in-out 2;
        }}
        .wheel {{
          transform-box: fill-box;
          transform-origin: center;
          animation: wheelSpin 2.2s linear infinite;
        }}
        .spark {{
          transform-box: fill-box;
          transform-origin: center;
          animation: floatStar 2.1s ease-in-out infinite;
        }}
      </style>
    </head>
    <body>
      <div id="stage" class="stage" onclick="triggerBurst()">
        <svg viewBox="0 0 400 210" width="100%" height="205" xmlns="http://www.w3.org/2000/svg">
          <ellipse cx="200" cy="188" rx="145" ry="11" fill="#5C3317" opacity="0.14"/>

          <!-- Floating brown & caramel stars -->
          <g class="spark" fill="#C86D3B">
            <polygon points="135,24 138,31 145,31 139,35 141,42 135,38 129,42 131,35 125,31 132,31"/>
            <polygon points="162,16 165,23 172,23 166,27 168,34 162,30 156,34 158,27 152,23 159,23"/>
            <polygon points="189,24 192,31 199,31 193,35 195,42 189,38 183,42 185,35 179,31 186,31"/>
          </g>

          <g class="mover">
            <!-- Friendly Espresso & Cream Robot -->
            <line x1="242" y1="42" x2="242" y2="62" stroke="#2B1810" stroke-width="3" stroke-linecap="round"/>
            <circle cx="242" cy="39" r="5.5" fill="#C86D3B"/>
            <rect x="205" y="62" width="74" height="56" rx="18" fill="#2B1810"/>
            <rect x="214" y="71" width="56" height="34" rx="10" fill="#FFFDF9"/>
            <circle cx="230" cy="87" r="4.5" fill="#7C4722"/>
            <circle cx="254" cy="87" r="4.5" fill="#7C4722"/>
            <path d="M234 96 Q242 103 250 96" stroke="#2B1810" stroke-width="2.5" fill="none" stroke-linecap="round"/>
            <rect x="216" y="122" width="52" height="28" rx="10" fill="#E6D7C3" stroke="#2B1810" stroke-width="2"/>

            <!-- Robot arm handling basket -->
            <path d="M216 135 L182 148" stroke="#2B1810" stroke-width="4" stroke-linecap="round"/>

            <!-- Shopping Basket -->
            <path d="M95 138 L186 138 L175 174 L106 174 Z" fill="#8C4A27" stroke="#2B1810" stroke-width="2.2"/>
            <path d="M108 138 Q140 108 172 138" stroke="#2B1810" stroke-width="2.5" fill="none"/>
            <line x1="118" y1="144" x2="122" y2="168" stroke="#FFFDF9" stroke-width="2" opacity="0.85"/>
            <line x1="140" y1="144" x2="140" y2="168" stroke="#FFFDF9" stroke-width="2" opacity="0.85"/>
            <line x1="162" y1="144" x2="158" y2="168" stroke="#FFFDF9" stroke-width="2" opacity="0.85"/>

            <!-- Animated Wheels -->
            <g class="wheel">
              <circle cx="116" cy="183" r="7.5" fill="#2B1810"/>
              <circle cx="116" cy="183" r="2.5" fill="#E6D7C3"/>
            </g>
            <g class="wheel">
              <circle cx="164" cy="183" r="7.5" fill="#2B1810"/>
              <circle cx="164" cy="183" r="2.5" fill="#E6D7C3"/>
            </g>
          </g>

          <!-- Warm Brown & Sage Trend Bars -->
          <rect x="302" y="140" width="12" height="36" rx="4" fill="#C9B7A3"/>
          <rect x="321" y="120" width="12" height="56" rx="4" fill="#9A6842"/>
          <rect x="340" y="96" width="12" height="80" rx="4" fill="#5C3317"/>
          <polyline points="300,114 326,92 353,66" fill="none" stroke="#C86D3B" stroke-width="3.2" stroke-linecap="round"/>
          <polygon points="347,64 359,62 356,74" fill="#C86D3B"/>
        </svg>
      </div>
      <script>
        const stage = document.getElementById("stage");
        function triggerBurst() {{
          stage.classList.remove("burst");
          void stage.offsetWidth;
          stage.classList.add("burst");
        }}
        if ({boost_js}) {{
          triggerBurst();
        }}
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=220, scrolling=False)


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
    "basket_boost": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown(
    """
    <style>
    @keyframes cacaoReveal {
        0% { opacity: 0; transform: translateY(14px); }
        100% { opacity: 1; transform: translateY(0px); }
    }
    .stApp {
        background: linear-gradient(180deg, #FBF8F4 0%, #F2EAE0 100%);
        color: #2B1810;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    .cacao-enter {
        animation: cacaoReveal 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    button[data-testid="stBaseButton-secondary"],
    .stButton > button[kind="secondary"] {
        background: #FFFDF9 !important;
        color: #2B1810 !important;
        border: 1px solid #D9C9B6 !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        transition: all 0.18s ease !important;
    }
    button[data-testid="stBaseButton-secondary"]:hover,
    .stButton > button[kind="secondary"]:hover {
        background: #EFE4D6 !important;
        border-color: #5C3317 !important;
        transform: translateY(-1px);
    }
    button[data-testid="stBaseButton-primary"],
    .stButton > button[kind="primary"] {
        background: #6E3B1E !important;
        color: #FFFDF9 !important;
        border: 1px solid #4E2812 !important;
        border-radius: 999px !important;
        font-weight: 700 !important;
        transition: all 0.18s ease !important;
    }
    button[data-testid="stBaseButton-primary"]:hover,
    .stButton > button[kind="primary"]:hover {
        background: #522A14 !important;
        transform: translateY(-1px);
    }
    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input {
        background-color: #FFFDF9 !important;
        color: #2B1810 !important;
        border: 1px solid #D6C5B2 !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #FFFDF9 !important;
        color: #2B1810 !important;
        border: 1.5px dashed #B9A288 !important;
        border-radius: 14px !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #2B1810 !important;
        color: #FFFDF9 !important;
        border: none !important;
        border-radius: 8px !important;
    }
    .brand-wordmark {
        font-size: 1.32rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #2B1810;
        padding-top: 6px;
        animation: cacaoReveal 0.4s ease-out both;
    }
    .brand-hero {
        font-size: 2.75rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #2B1810;
        padding-top: 26px;
        animation: cacaoReveal 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    .surface-card {
        background: #FFFDF9;
        border: 1px solid #E5D8C7;
        border-radius: 18px;
        padding: 20px;
        margin-top: 12px;
        box-shadow: 0 6px 20px rgba(43, 24, 16, 0.05);
        animation: cacaoReveal 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    .metric-high {
        background: #FDECE8;
        border: 1px solid #E9B6AA;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-risk {
        background: #FDF4E3;
        border: 1px solid #E8CFA2;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-low {
        background: #EEF6EE;
        border: 1px solid #BBD8BC;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-neutral {
        background: #F6EFE6;
        border: 1px solid #DFD0BE;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 700;
        color: #4A3224;
        margin-bottom: 6px;
    }
    .metric-num {
        font-size: 1.75rem;
        font-weight: 800;
        color: #2B1810;
        line-height: 1.15;
    }
    .cat-card {
        background: #F8F2EA;
        border: 1px solid #E3D5C3;
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 10px;
        transition: transform 0.16s ease, border-color 0.16s ease;
    }
    .cat-card:hover {
        transform: translateY(-2px);
        border-color: #8C4A27;
    }
    .cat-header {
        display: flex;
        justify-content: space-between;
        font-weight: 700;
        color: #2B1810;
        margin-bottom: 4px;
    }
    .cat-score {
        font-weight: 800;
        color: #8C4A27;
    }
    .cat-desc {
        font-size: 0.86rem;
        color: #564134;
        line-height: 1.45;
    }
    .info-box {
        background: #F8F2EA;
        border: 1px solid #E3D5C3;
        border-radius: 14px;
        padding: 16px;
        margin-top: 8px;
        color: #2B1810;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

b_col, n1, n2, n3, n4, _ = st.columns([1.8, 1.0, 1.0, 1.3, 1.0, 2.5])
with b_col:
    st.markdown('<div class="brand-wordmark">RetainCraft</div>', unsafe_allow_html=True)
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

st.markdown("---")

if st.session_state.active_tab == "About":
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    st.subheader("RetainCraft")
    st.write("Review triage, category scorecards, retention playbooks, and AI workflows.")
    st.markdown("</div>", unsafe_allow_html=True)
elif st.session_state.active_tab == "How It Works":
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    st.subheader("RetainCraft")
    st.write("1. Paste reviews or upload any CSV.\n2. Click Analyze Reviews.\n3. Explore scorecards, playbooks, drafts, copilot, and photo inspection.")
    st.markdown("</div>", unsafe_allow_html=True)
elif st.session_state.active_tab == "Insights":
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    st.subheader("RetainCraft")
    if st.session_state.scored_table is not None and not st.session_state.scored_table.empty:
        if st.session_state.get("genai_insights"):
            st.markdown(st.session_state.genai_insights)
        if "root_cause_theme" in st.session_state.scored_table.columns:
            st.bar_chart(st.session_state.scored_table["root_cause_theme"].value_counts(), color="#8C4A27")
        st.dataframe(st.session_state.scored_table, width="stretch")
        st.download_button(
            label="Download Scored CSV",
            data=st.session_state.scored_table.to_csv(index=False).encode("utf-8"),
            file_name="retaincraft_scored_output.csv",
            mime="text/csv",
        )
    else:
        st.info("Upload a CSV or paste reviews on Home and click Analyze Reviews.")
    st.markdown("</div>", unsafe_allow_html=True)
else:
    hero_left, hero_right = st.columns([1.2, 1.0], gap="large")
    with hero_left:
        st.markdown('<div class="brand-hero">RetainCraft</div>', unsafe_allow_html=True)
        if st.button("Animate Basket", key="btn_animate_basket"):
            st.session_state.basket_boost = not st.session_state.basket_boost
    with hero_right:
        render_robot_basket_widget(boost=bool(st.session_state.basket_boost))
        st.session_state.basket_boost = False

    left_box, right_box = st.columns([1.0, 1.2], gap="large")

    with left_box:
        st.markdown('<div class="surface-card">', unsafe_allow_html=True)
        pasted_text = st.text_area(
            "Reviews",
            height=155,
            placeholder="Paste reviews (one per line)...",
            label_visibility="collapsed",
        )
        uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded_csv is not None:
            df_loaded = pd.read_csv(uploaded_csv)
            st.session_state.uploaded_df = df_loaded
            st.session_state.uploaded_name = uploaded_csv.name
            st.caption(f"{uploaded_csv.name} ({len(df_loaded)} rows)")

        if st.session_state.scored_table is not None and not st.session_state.scored_table.empty:
            df_s = st.session_state.scored_table
            top_cat = str(df_s["category"].value_counts().index[0]) if "category" in df_s.columns else "—"
            top_city = str(df_s["city"].value_counts().index[0]) if "city" in df_s.columns else "—"
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown(f"**Rows:** {fmt_int(len(df_s))}  |  **Top Category:** {top_cat}  |  **Top Region:** {top_city}")
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_box:
        st.markdown('<div class="surface-card">', unsafe_allow_html=True)
        analyze_clicked = st.button("Analyze Reviews", type="primary", use_container_width=True)

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
                    st.warning("Paste reviews or upload a CSV file first.")

            if scored_df is not None and summary is not None:
                st.session_state.scored_table = scored_df
                st.session_state.scored_summary = summary
                st.session_state.genai_insights = generate_genai_insights(scored_df, summary, batch_label)
                st.session_state.playbook_text = generate_retention_playbook(scored_df, summary)
                st.session_state.reply_draft = None
                st.session_state.copilot_answer = None
                st.session_state.basket_boost = True

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
                f'<div class="metric-high"><div class="metric-label">High Urgency</div><div class="metric-num">{fmt_int(high_val)}</div></div>',
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f'<div class="metric-risk"><div class="metric-label">At Risk</div><div class="metric-num">{fmt_int(at_risk_val)}</div></div>',
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(
                f'<div class="metric-low"><div class="metric-label">Low Risk</div><div class="metric-num">{fmt_int(low_risk_val)}</div></div>',
                unsafe_allow_html=True,
            )

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f'<div class="metric-neutral"><div class="metric-label">At-Risk Revenue</div><div class="metric-num">{fmt_inr(at_risk_rev_val)}</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-neutral"><div class="metric-label">Avg Churn Risk</div><div class="metric-num">{fmt_pct(avg_churn_val)}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        if st.session_state.get("genai_insights"):
            st.markdown(st.session_state.genai_insights)
        elif st.session_state.get("scored_summary"):
            st.write(
                f"- High Urgency: {fmt_int(high_val)} | At Risk: {fmt_int(at_risk_val)} | "
                f"At-Risk Revenue: {fmt_inr(at_risk_rev_val)} | Goodwill Pool: {fmt_inr(total_goodwill_val)}"
            )
        else:
            st.caption("—")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.scored_table is not None and not st.session_state.scored_table.empty:
        df_scored = st.session_state.scored_table

        row_a_left, row_a_right = st.columns([1.1, 1.0], gap="large")
        with row_a_left:
            st.markdown('<div class="surface-card">', unsafe_allow_html=True)
            if "root_cause_theme" in df_scored.columns:
                st.bar_chart(df_scored["root_cause_theme"].value_counts(), color="#8C4A27")
            st.markdown("</div>", unsafe_allow_html=True)
        with row_a_right:
            st.markdown('<div class="surface-card">', unsafe_allow_html=True)
            strengths, complaints = build_strengths_and_complaints(df_scored)
            s_col, c_col = st.columns(2)
            with s_col:
                st.markdown("**Strengths**")
                for item in strengths:
                    st.markdown(f"- {item}")
            with c_col:
                st.markdown("**Complaints**")
                for item in complaints:
                    st.markdown(f"- {item}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="surface-card">', unsafe_allow_html=True)
        cards = build_category_scorecards(df_scored)
        for idx in range(0, len(cards), 2):
            c_left, c_right = st.columns(2, gap="medium")
            for col_container, card in [(c_left, cards[idx]), (c_right, cards[idx + 1] if idx + 1 < len(cards) else None)]:
                if card:
                    with col_container:
                        st.markdown(
                            f"""
                            <div class="cat-card">
                                <div class="cat-header"><span>{card['theme']}</span><span class="cat-score">{card['score']}</span></div>
                                <div class="cat-desc">{card['summary']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
        st.markdown("</div>", unsafe_allow_html=True)

        row_c_left, row_c_right = st.columns(2, gap="large")
        with row_c_left:
            st.markdown('<div class="surface-card">', unsafe_allow_html=True)
            st.markdown("**Retention Playbook**")
            if st.session_state.get("playbook_text"):
                st.markdown(st.session_state.playbook_text)
            st.markdown("</div>", unsafe_allow_html=True)
        with row_c_right:
            st.markdown('<div class="surface-card">', unsafe_allow_html=True)
            st.markdown("**Reply & Escalation Drafter**")
            ticket_options = [
                f"{row['review_id']} | {row['urgency_class']} | {str(row['review_text'])[:42]}"
                for _, row in df_scored.head(50).iterrows()
            ]
            selected_ticket = st.selectbox("Ticket", ticket_options, label_visibility="collapsed")
            if st.button("Draft Response", key="btn_draft_reply"):
                sel_idx = ticket_options.index(selected_ticket)
                st.session_state.reply_draft = draft_customer_and_escalation(df_scored.iloc[sel_idx].to_dict())
            if st.session_state.get("reply_draft"):
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown(st.session_state.reply_draft)
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        row_d_left, row_d_right = st.columns(2, gap="large")
        with row_d_left:
            st.markdown('<div class="surface-card">', unsafe_allow_html=True)
            st.markdown("**Ask Your Data**")
            copilot_q = st.text_input("Question", placeholder="Ask about this batch...", label_visibility="collapsed")
            if st.button("Ask Copilot", key="btn_ask_copilot"):
                if copilot_q.strip():
                    st.session_state.copilot_answer = ask_data_copilot(
                        copilot_q.strip(), df_scored, st.session_state.scored_summary or {}
                    )
                else:
                    st.warning("Enter a question.")
            if st.session_state.get("copilot_answer"):
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown(st.session_state.copilot_answer)
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with row_d_right:
            st.markdown('<div class="surface-card">', unsafe_allow_html=True)
            st.markdown("**Defect Photo Inspector**")
            defect_photo = st.file_uploader(
                "Photo", type=["png", "jpg", "jpeg", "webp"], key="defect_photo_uploader", label_visibility="collapsed"
            )
            defect_note = st.text_input("Context", placeholder="Defect context...", label_visibility="collapsed")
            if st.button("Inspect Photo", key="btn_inspect_photo"):
                if defect_photo is not None:
                    st.session_state.defect_report = inspect_defect_photo_with_gemini(
                        defect_photo.getvalue(), defect_photo.type or "image/jpeg", defect_note
                    )
                else:
                    st.warning("Upload an image first.")
            if st.session_state.get("defect_report"):
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown(st.session_state.defect_report)
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="surface-card">', unsafe_allow_html=True)
        st.dataframe(df_scored.head(25), width="stretch")
        st.download_button(
            label="Download Full Scored CSV",
            data=df_scored.to_csv(index=False).encode("utf-8"),
            file_name="retaincraft_scored_output.csv",
            mime="text/csv",
        )
        st.markdown("</div>", unsafe_allow_html=True)