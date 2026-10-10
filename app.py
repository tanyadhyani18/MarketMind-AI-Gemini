import hashlib
import pandas as pd
import plotly.express as px
import streamlit as st
from marketmind_agents import (REQUIRED, prepare_data, summarize, make_large_sample, orchestrate)

st.set_page_config(page_title="MarketMind AI", page_icon="📊", layout="wide")
st.title("📊 MarketMind AI")
st.caption("Agentic Marketing Intelligence | MBA Marketing + Big Data Analytics")

SMALL = pd.DataFrame({
    "Campaign": ["Instagram Ads", "Google Ads", "Facebook Ads", "Email Marketing"],
    "Channel": ["Social", "Search", "Social", "Email"],
    "Impressions": [50000, 40000, 60000, 20000], "Clicks": [2500, 3200, 1800, 1500],
    "Conversions": [100, 240, 54, 180], "Spend": [10000, 15000, 8000, 2000],
    "Revenue": [30000, 60000, 15000, 25000]})
LARGE = make_large_sample()


def secret(name, default=""):
    try: return st.secrets.get(name, default)
    except Exception: return default


# ------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("How to use")
    st.write("1. Upload a campaign CSV, or pick a sample.\n2. (Optional) add a free Gemini key for AI mode.\n3. Click **Run AI agents**.")
    sample = st.radio("Sample data (used when nothing is uploaded)",
                      ["Small: 4 campaigns", "Large: 7 campaigns × 6 months"])
    st.download_button("Download small sample CSV", SMALL.to_csv(index=False).encode(), "marketmind_sample_campaigns.csv", "text/csv")
    st.download_button("Download large sample CSV", LARGE.to_csv(index=False).encode(), "marketmind_sample_large.csv", "text/csv")
    st.divider()
    st.header("AI mode (optional)")
    api_key = secret("LLM_API_KEY") or st.text_input("API key (Gemini free key works)", type="password",
                                                      help="Get a free key at aistudio.google.com. It is used only for this session.")
    with st.expander("Advanced"):
        base_url = st.text_input("Base URL", secret("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"))
        model = st.text_input("Model", secret("MODEL_NAME", "gemini-3.8-flash"))
    st.caption("Without a key the app uses transparent Python rules. With a key, the Strategist and Manager agents use an LLM, "
               "but numbers are always computed by Python and checked by the Manager.")

# ------------------------------------------------------------- data
upload = st.file_uploader("Upload campaign CSV", type=["csv"])
try:
    raw = pd.read_csv(upload) if upload else (SMALL if sample.startswith("Small") else LARGE).copy()
    df = prepare_data(raw)
except Exception as e:
    st.error(f"Could not analyse file: {e}")
    st.info("Required columns: " + ", ".join(REQUIRED) + " (optional: Date)")
    st.stop()
summary = summarize(df)

# ------------------------------------------------------------- run agents
sig = hashlib.md5((summary.to_csv() + model).encode()).hexdigest()
rule_result = orchestrate(summary)                       # cheap, always available
run = st.button("🚀 Run AI agents", type="primary", disabled=not api_key,
                help=None if api_key else "Add an API key in the sidebar to use the LLM agents.")
if run:
    with st.spinner("Strategist and Manager agents are working (can take up to a minute on the free tier)..."):
        st.session_state["ai"] = (sig, orchestrate(summary, {"api_key": api_key, "base_url": base_url, "model": model}))
ai = st.session_state.get("ai")
res = ai[1] if ai and ai[0] == sig else rule_result
m, p, recs, review = res["metrics"], res["perf"], res["recs"], res["review"]

st.caption(f"Analysing {len(summary)} campaigns ({len(df)} rows)")
c = st.columns(4)
c[0].metric("Total ad spend", f"₹{m['total_spend']:,.0f}")
c[1].metric("Attributed revenue", f"₹{m['total_revenue']:,.0f}")
c[2].metric("Conversions", f"{m['conversions']:,}")
c[3].metric("Overall ROAS", f"{m['overall_roas']:.2f}x")
st.caption("ROAS = attributed revenue ÷ ad spend; it is not profit.")

t1, t2, t3, t4 = st.tabs(["📈 Dashboard", "🤖 Agent workflow", "💬 Customer feedback", "📄 Data & reports"])
with t1:
    l, r = st.columns(2)
    l.plotly_chart(px.bar(summary, x="Campaign", y="Conversions", color="Channel", title="Conversions by campaign"), width="stretch")
    r.plotly_chart(px.bar(summary, x="Campaign", y="ROAS", color="Channel", title="ROAS by campaign"), width="stretch")
    l2, r2 = st.columns(2)
    l2.plotly_chart(px.bar(summary.groupby("Channel", as_index=False)["Spend"].sum(), x="Channel", y="Spend", color="Channel", title="Spend by channel"), width="stretch")
    r2.plotly_chart(px.scatter(summary, x="Spend", y="Revenue", size="Conversions", color="Campaign", title="Spend vs revenue (bubble = conversions)"), width="stretch")
    if "Date" in df.columns:
        mon = df.assign(Month=df["Date"].dt.to_period("M").astype(str)).groupby(["Month", "Campaign"], as_index=False)[["Spend", "Revenue"]].sum()
        mon["ROAS"] = (mon["Revenue"] / mon["Spend"]).round(2)
        st.plotly_chart(px.line(mon, x="Month", y="ROAS", color="Campaign", markers=True, title="ROAS trend by month"), width="stretch")
    st.dataframe(summary, width="stretch", hide_index=True)

with t2:
    st.info(f"Plan written by: **{res['source']}**" + (" · add a key in the sidebar and click Run AI agents for LLM mode" if res["source"] == "Rules" and not api_key else ""))
    st.subheader("Agent 1: Data Analyst")
    a = st.columns(4)
    a[0].metric("Overall CTR", f"{m['overall_ctr_pct']:.2f}%")
    a[1].metric("Conversion rate", f"{m['overall_conversion_rate_pct']:.2f}%")
    a[2].metric("Cost per conversion", f"₹{m['overall_cpa']:,.2f}" if m["overall_cpa"] else "n/a")
    a[3].metric("Campaigns", m["campaign_count"])
    st.subheader("Agent 2: Campaign Performance Analyst")
    st.write(f"Highest ROAS: **{p['best_campaign']}** ({p['best_roas']:.2f}x) · Lowest ROAS: **{p['weakest_campaign']}** ({p['weakest_roas']:.2f}x)")
    if p["funnel_leak"]:
        f = p["funnel_leak"]
        st.write(f"Funnel leak: **{f['campaign']}** gets clicks (CTR {f['ctr_pct']:.2f}%) but converts below average ({f['conversion_rate_pct']:.2f}%).")
    st.dataframe(summary[["Campaign", "Channel", "Spend", "Revenue", "ROAS", "CTR (%)", "Conversion Rate (%)", "CPA"]], width="stretch", hide_index=True)
    st.subheader("Agent 3: Marketing Strategist")
    st.dataframe(pd.DataFrame(recs), width="stretch", hide_index=True)
    st.subheader("Agent 4: Manager (reviewer)")
    st.success("Approved by Manager") if review["approved"] else st.error("Not approved: a human marketer must review this plan")
    for k, ok in review["checklist"].items(): st.write(("✅ " if ok else "❌ ") + k)
    st.write(f"Revision rounds used: **{res['revisions']}**")
    if review.get("llm_feedback"): st.info("LLM reviewer: " + review["llm_feedback"])
    st.warning("Recommendations are decision support, not decisions. Review business context first. The app never changes ad budgets.")
    with st.expander("Agent activity log"):
        for e in res["events"]: st.write("- " + e)

with t3:
    st.write("Upload a CSV with a `Review` column for approximate keyword-based sentiment.")
    review_file = st.file_uploader("Upload customer reviews CSV", type=["csv"], key="reviews")
    if review_file:
        try:
            reviews = pd.read_csv(review_file)
            if "Review" not in reviews.columns: st.error("CSV needs a column named Review.")
            else:
                pos = {"good", "great", "love", "excellent", "easy", "fast", "helpful", "happy", "recommend"}
                neg = {"bad", "poor", "slow", "expensive", "broken", "late", "difficult", "refund", "disappointed"}
                def sentiment(v):
                    w = set(str(v).lower().replace(".", " ").replace(",", " ").split())
                    sc = len(w & pos) - len(w & neg)
                    return "Positive" if sc > 0 else "Negative" if sc < 0 else "Neutral / unclear"
                reviews["Approximate sentiment"] = reviews["Review"].fillna("").map(sentiment)
                st.dataframe(reviews, width="stretch", hide_index=True)
                st.plotly_chart(px.histogram(reviews, x="Approximate sentiment", title="Approximate customer sentiment"), width="stretch")
                st.download_button("Download sentiment CSV", reviews.to_csv(index=False).encode(), "marketmind_sentiment.csv", "text/csv")
        except Exception as e: st.error(str(e))
    else: st.info("Optional section. Keyword sentiment is approximate, not an LLM or trained model.")

with t4:
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Download analysed CSV", df.to_csv(index=False).encode(), "marketmind_analysis.csv", "text/csv")
    report = f"""# MarketMind AI Report

Campaigns analysed: {m['campaign_count']}
Total spend: ₹{m['total_spend']:,.0f}
Attributed revenue: ₹{m['total_revenue']:,.0f}
Conversions: {m['conversions']}
Overall ROAS: {m['overall_roas']:.2f}x | CTR: {m['overall_ctr_pct']:.2f}% | Conversion rate: {m['overall_conversion_rate_pct']:.2f}%

Highest ROAS: {p['best_campaign']} ({p['best_roas']:.2f}x)
Lowest ROAS: {p['weakest_campaign']} ({p['weakest_roas']:.2f}x)

## Recommendations (written by: {res['source']})
""" + "\n".join(f"- **{r['Priority']}**: {r['Action']} Evidence: {r['Evidence']} KPI: {r['KPI to track']}. Budget shift: ₹{r['Budget shift (₹)']:,}. Impact: {r['Expected impact']}" for r in recs) + f"""

## Manager review
Status: {'Approved' if review['approved'] else 'NOT approved, human review needed'} (revision rounds: {res['revisions']})
""" + "\n".join(f"- {'PASS' if ok else 'FAIL'}: {k}" for k, ok in review["checklist"].items())
    st.download_button("Download report (Markdown)", report, "marketmind_report.md", "text/markdown")
st.divider()
st.caption("Educational prototype. Sample data is illustrative. Validate insights before taking action.")
