"""MarketMind AI: the agents (logic only, no Streamlit code in here).

Agents
  1. Data Analyst          -> portfolio numbers (pure Python, always exact)
  2. Performance Analyst   -> best / weakest / funnel-leak campaigns (pure Python)
  3. Marketing Strategist  -> recommendations. LLM if an API key is given, otherwise rules.
  4. Manager (Critic)      -> checks the plan (automatic checks + optional LLM review),
                              and sends it back for ONE revision if it fails.
The LLM never calculates numbers: it only receives numbers already computed by agents 1-2,
and the Manager checks that every number in the plan can be traced back to the data.
"""
import json
import re
import time
import numpy as np
import pandas as pd

REQUIRED = ["Campaign", "Channel", "Impressions", "Clicks", "Conversions", "Spend", "Revenue"]
NUMERIC = ["Impressions", "Clicks", "Conversions", "Spend", "Revenue"]
REC_KEYS = ["Priority", "Action", "Campaign", "Budget shift (₹)", "Evidence", "KPI to track", "Expected impact"]


# ------------------------------------------------------------------ data
def add_kpis(df):
    df = df.copy()
    clicks = df["Clicks"].replace(0, np.nan)
    conv = df["Conversions"].replace(0, np.nan)
    df["CTR (%)"] = (df["Clicks"] / df["Impressions"] * 100).round(2)
    df["Conversion Rate (%)"] = (df["Conversions"] / clicks * 100).round(2)
    df["CPC"] = (df["Spend"] / clicks).round(2)
    df["CPA"] = (df["Spend"] / conv).round(2)
    df["ROAS"] = (df["Revenue"] / df["Spend"]).round(2)
    return df


def prepare_data(df):
    """Validate and add KPIs. Optional 'Date' column enables the trend chart."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing: raise ValueError("Missing columns: " + ", ".join(missing))
    if df.empty: raise ValueError("CSV has no rows.")
    df = df.copy()
    for c in NUMERIC: df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[NUMERIC].isna().any().any(): raise ValueError("Metric columns must contain valid numbers.")
    if (df[NUMERIC] < 0).any().any(): raise ValueError("Metrics cannot be negative.")
    if (df["Impressions"] <= 0).any() or (df["Spend"] <= 0).any():
        raise ValueError("Impressions and spend must be above zero.")
    if (df["Clicks"] > df["Impressions"]).any() or (df["Conversions"] > df["Clicks"]).any():
        raise ValueError("Clicks cannot exceed impressions; conversions cannot exceed clicks.")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if df["Date"].isna().any(): raise ValueError("Date column must contain valid dates, e.g. 2026-04-30.")
    return add_kpis(df)


def summarize(df):
    """One row per campaign (sums repeated rows such as monthly data), KPIs recalculated."""
    g = df.groupby(["Campaign", "Channel"], as_index=False)[NUMERIC].sum()
    return add_kpis(g).sort_values("ROAS", ascending=False).reset_index(drop=True)


def make_large_sample():
    """Bigger practice dataset: 7 campaigns x 6 months = 42 rows (illustrative, not real)."""
    rng = np.random.default_rng(7)
    # name, channel, impressions/month, CTR, conversion rate, spend/month, avg order value, monthly CVR drift
    camps = [("Instagram Ads", "Social", 120000, .032, .040, 26000, 600, 0.00),
             ("Facebook Ads", "Social", 150000, .020, .018, 18000, 600, -0.04),
             ("Google Search", "Search", 70000, .065, .060, 45000, 700, 0.02),
             ("YouTube Ads", "Video", 200000, .012, .015, 15000, 600, 0.00),
             ("Email Marketing", "Email", 30000, .060, .040, 6000, 650, 0.01),
             ("WhatsApp Broadcast", "Messaging", 25000, .120, .030, 9000, 550, 0.00),
             ("Influencer Collab", "Influencer", 180000, .028, .022, 40000, 800, 0.00)]
    dates = ["2026-04-30", "2026-05-31", "2026-06-30", "2026-07-31", "2026-08-31", "2026-09-30"]
    rows = []
    for name, ch, imp, ctr, cvr, spend, aov, drift in camps:
        for i, d in enumerate(dates):
            im = int(imp * (1 + rng.normal(0, .08)))
            cl = int(im * ctr * (1 + rng.normal(0, .06)))
            cv = int(cl * cvr * (1 + drift * i) * (1 + rng.normal(0, .08)))
            rows.append(dict(Date=d, Campaign=name, Channel=ch, Impressions=im, Clicks=cl, Conversions=cv,
                             Spend=int(spend * (1 + rng.normal(0, .05))),
                             Revenue=int(cv * aov * (1 + rng.normal(0, .05)))))
    return pd.DataFrame(rows)


# --------------------------------------------------------- agents 1 and 2
def data_analyst_agent(s):
    spend, rev = float(s["Spend"].sum()), float(s["Revenue"].sum())
    clicks, imp, conv = float(s["Clicks"].sum()), float(s["Impressions"].sum()), float(s["Conversions"].sum())
    return {"campaign_count": int(len(s)), "total_spend": spend, "total_revenue": rev,
            "conversions": int(conv), "overall_roas": round(rev / spend, 2),
            "overall_ctr_pct": round(clicks / imp * 100, 2),
            "overall_conversion_rate_pct": round(conv / clicks * 100, 2) if clicks else 0.0,
            "overall_cpa": round(spend / conv, 2) if conv else None}


def performance_analyst_agent(s, m):
    best, weak = s.loc[s["ROAS"].idxmax()], s.loc[s["ROAS"].idxmin()]
    leaks = s[(s["CTR (%)"] >= s["CTR (%)"].median()) & (s["Conversion Rate (%)"] < m["overall_conversion_rate_pct"])]
    leak = leaks.sort_values("CTR (%)", ascending=False).iloc[0] if len(leaks) else None
    costly = s.loc[s["CPA"].idxmax()] if s["CPA"].notna().any() else best
    return {"best_campaign": str(best["Campaign"]), "best_roas": float(best["ROAS"]),
            "weakest_campaign": str(weak["Campaign"]), "weakest_roas": float(weak["ROAS"]),
            "weakest_spend": float(weak["Spend"]),
            "funnel_leak": None if leak is None else {"campaign": str(leak["Campaign"]),
                "ctr_pct": float(leak["CTR (%)"]), "conversion_rate_pct": float(leak["Conversion Rate (%)"])},
            "highest_cpa_campaign": str(costly["Campaign"]), "highest_cpa": float(costly["CPA"])}


# ------------------------------------------------- agent 3: rule-based plan
def rule_strategist(m, p, s):
    recs = []
    if p["best_campaign"] != p["weakest_campaign"]:
        shift = int(round(0.15 * p["weakest_spend"], -2)) or int(round(0.15 * p["weakest_spend"]))
        gain = shift * (p["best_roas"] - p["weakest_roas"])
        recs.append({"Priority": "1 — Reallocate",
            "Action": f"Move part of the {p['weakest_campaign']} budget to {p['best_campaign']} as a controlled test.",
            "Campaign": p["weakest_campaign"], "Budget shift (₹)": shift,
            "Evidence": f"{p['weakest_campaign']} ROAS is {p['weakest_roas']:.2f}x versus {p['best_campaign']} at {p['best_roas']:.2f}x.",
            "KPI to track": "ROAS, CPA, conversion volume",
            "Expected impact": f"Illustrative upside of about ₹{gain:,.0f} revenue if {p['best_campaign']} keeps its ROAS (returns usually diminish as spend grows)."})
    lk = p["funnel_leak"]
    if lk:
        recs.append({"Priority": "2 — Fix funnel",
            "Action": f"Improve the landing page and offer for {lk['campaign']}: people click but few convert.",
            "Campaign": lk["campaign"], "Budget shift (₹)": 0,
            "Evidence": f"CTR is {lk['ctr_pct']:.2f}% (above median) but conversion rate is {lk['conversion_rate_pct']:.2f}% versus {m['overall_conversion_rate_pct']:.2f}% for the portfolio.",
            "KPI to track": "Conversion rate, CPA",
            "Expected impact": "More conversions from the same media spend if the gap to the portfolio rate narrows."})
    else:
        recs.append({"Priority": "2 — Fix cost",
            "Action": f"Audit targeting and creative for {p['highest_cpa_campaign']}, which has the highest cost per conversion.",
            "Campaign": p["highest_cpa_campaign"], "Budget shift (₹)": 0,
            "Evidence": f"CPA is ₹{p['highest_cpa']:,.2f} versus ₹{m['overall_cpa']:,.2f} for the portfolio." if m["overall_cpa"] else "Highest CPA in the portfolio.",
            "KPI to track": "CPA, conversion rate", "Expected impact": "Lower cost per conversion."})
    recs.append({"Priority": "3 — Learn",
        "Action": f"Document what drives {p['best_campaign']} (audience, creative, offer) and test it on other channels.",
        "Campaign": p["best_campaign"], "Budget shift (₹)": 0,
        "Evidence": f"{p['best_campaign']} has the highest ROAS at {p['best_roas']:.2f}x.",
        "KPI to track": "ROAS, audience quality", "Expected impact": "Repeatable playbook for the other campaigns."})
    recs.append({"Priority": "4 — Validate",
        "Action": "Before moving any budget, validate margin, attribution, audience overlap and campaign goals with an incremental test.",
        "Campaign": "Portfolio", "Budget shift (₹)": 0,
        "Evidence": f"Overall ROAS is {m['overall_roas']:.2f}x; ROAS is not profit and does not prove causation.",
        "KPI to track": "Contribution margin, incremental conversions", "Expected impact": "Avoids scaling a campaign that only looks good on paper."})
    return recs


# ---------------------------------------------------------- agent 4: manager
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text):
    out = []
    for t in _NUM.findall(text):
        try: out.append(float(t.replace(",", "")))
        except ValueError: pass
    return out


def grounded_check(recs, s, m, p):
    """Share of numbers in Action/Evidence that exist in the computed data."""
    pool = []
    for v in s.select_dtypes("number").to_numpy().ravel(): pool.append(float(v))
    pool += [float(v) for v in m.values() if isinstance(v, (int, float))]
    pool += [float(v) for v in p.values() if isinstance(v, (int, float))]
    if p["funnel_leak"]: pool += list(p["funnel_leak"].values())[1:]
    allowed = {round(v / k, 3) for v in pool if v == v for k in (1, 1e3, 1e5)}
    text = " ".join(f"{r.get('Action', '')} {r.get('Evidence', '')}" for r in recs)
    claims = [c for c in _numbers(text) if not (c == int(c) and c <= 30)]
    ok = sum(any(abs(c - a) <= max(0.01 * abs(a), 0.06) for a in allowed) for c in claims)
    return ok, len(claims)


def rule_review(recs, s, m, p):
    names = set(s["Campaign"]) | {"Portfolio"}
    shifts = 0.0
    for r in recs:
        try: shifts += max(0.0, float(r.get("Budget shift (₹)") or 0))
        except (TypeError, ValueError): pass
    ok, total = grounded_check(recs, s, m, p)
    text = " ".join(f"{r.get('Action', '')} {r.get('Evidence', '')}" for r in recs)
    checks = {
        "At least 3 actions": len(recs) >= 3,
        "Every action names a real campaign (or Portfolio)": all(r.get("Campaign") in names for r in recs),
        "Every action has a KPI to track": all(str(r.get("KPI to track", "")).strip() for r in recs),
        "Budget moved is realistic (max 20% of total spend)": shifts <= 0.20 * m["total_spend"],
        f"Numbers traceable to the data ({ok} of {total})": total == 0 or ok / total >= 0.8,
        "Includes a validation step (margin / attribution / test)": bool(re.search(r"margin|attribution|validate|incremental|test", text, re.I)),
    }
    failed = [k for k, v in checks.items() if not v]
    return {"checklist": checks, "approved": not failed,
            "feedback": ("Fix these: " + "; ".join(failed)) if failed else ""}


# ------------------------------------------------------------- LLM helpers
def default_llm(cfg):
    """OpenAI-compatible client (works with Gemini's free endpoint, OpenAI, Groq...)."""
    from openai import OpenAI
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg.get("base_url") or None)

    def call(system, user, temperature=0.3):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        use_temp = True
        for attempt in range(4):
            try:
                kw = dict(model=cfg["model"], messages=msgs)
                if use_temp: kw["temperature"] = temperature
                return client.chat.completions.create(**kw).choices[0].message.content or ""
            except Exception as e:
                msg = str(e).lower()
                if use_temp and "temperature" in msg: use_temp = False
                elif any(k in msg for k in ("429", "rate", "quota", "resource_exhausted")): time.sleep(10 * (attempt + 1))
                else: raise
        raise RuntimeError("Rate limit reached. Wait a minute and try again.")
    return call


def _json(text, kind):
    m = re.search(r"\[.*\]" if kind == "list" else r"\{.*\}", text, re.S)
    if not m: raise ValueError("LLM did not return JSON")
    return json.loads(m.group(0))


def llm_strategist(call, s, m, p, feedback=""):
    table = s[["Campaign", "Channel", "Spend", "Revenue", "ROAS", "CTR (%)", "Conversion Rate (%)", "CPA"]].to_dict("records")
    system = ("You are the Head of Marketing Strategy at a D2C brand, trained in MBA frameworks (ROAS, CPA, funnel analysis, "
              "budget allocation). Use ONLY numbers from the data provided. Never invent statistics.")
    user = (f"CAMPAIGN DATA (per campaign):\n{json.dumps(table, default=str)}\n\nPORTFOLIO METRICS:\n{json.dumps(m, default=str)}\n\n"
            f"PERFORMANCE FINDINGS:\n{json.dumps(p, default=str)}\n"
            + (f"\nMANAGER FEEDBACK TO FIX:\n{feedback}\n" if feedback else "") +
            "\nReturn ONLY a JSON list of 4 recommendations (no markdown). Each item has keys: "
            '"Priority" (e.g. "1 — Reallocate"), "Action" (specific, names a campaign), "Campaign" (an exact campaign name from the data, '
            'or "Portfolio"), "Budget shift (₹)" (number; 0 if none; total shifts at most 20% of total spend), "Evidence" (quote numbers '
            'from the data), "KPI to track", "Expected impact" (state assumptions). Put rupee amounts ONLY in "Budget shift (₹)" and '
            '"Expected impact", never in Action or Evidence. Include one recommendation about validating margin/attribution before scaling.')
    items = _json(call(system, user, 0.4), "list")
    recs = []
    for it in items[:6]:
        r = {k: it.get(k, "") for k in REC_KEYS}
        try: r["Budget shift (₹)"] = int(float(r["Budget shift (₹)"] or 0))
        except (TypeError, ValueError): r["Budget shift (₹)"] = 0
        recs.append(r)
    if not recs: raise ValueError("LLM returned no recommendations")
    return recs


def llm_reviewer(call, recs, s):
    system = "You are a strict Marketing Director reviewing a campaign plan. Judge only against the data given."
    user = (f"CAMPAIGNS: {json.dumps(s[['Campaign', 'ROAS', 'Spend']].to_dict('records'), default=str)}\n\nPLAN:\n{json.dumps(recs, default=str)}\n\n"
            "Is the plan specific, consistent with the data, realistic and useful? Reply ONLY with JSON: "
            '{"approved": true/false, "feedback": "specific fixes if not approved, else empty"}')
    r = _json(call(system, user, 0.2), "dict")
    return bool(r.get("approved")), str(r.get("feedback", ""))


def review_plan(recs, s, m, p, call, events):
    rv = rule_review(recs, s, m, p)
    rv["llm_feedback"] = ""
    if call:
        try:
            ok, fb = llm_reviewer(call, recs, s)
            rv["llm_feedback"] = fb
            rv["checklist"]["Manager (LLM) judges the plan useful and consistent"] = ok
            if not ok: rv["approved"] = False; rv["feedback"] = (rv["feedback"] + " " + fb).strip()
        except Exception as e:
            events.append(f"LLM reviewer unavailable ({e}); automatic checks only.")
    return rv


# ------------------------------------------------------------ orchestration
def orchestrate(s, cfg=None, llm=None):
    """Run the whole crew. cfg = dict(api_key, base_url, model) enables LLM mode; `llm` lets tests inject a fake."""
    events = []
    m = data_analyst_agent(s); events.append("Agent 1 (Data Analyst) computed portfolio metrics.")
    p = performance_analyst_agent(s, m); events.append("Agent 2 (Performance Analyst) ranked campaigns and found funnel leaks.")
    call = llm or (default_llm(cfg) if cfg else None)
    source, revisions = "Rules", 0
    if call:
        try:
            recs = llm_strategist(call, s, m, p); source = "LLM"
            events.append("Agent 3 (Strategist) wrote the plan using the LLM.")
        except Exception as e:
            events.append(f"Agent 3: LLM failed ({e}). Used rule-based plan instead."); recs = rule_strategist(m, p, s)
    else:
        recs = rule_strategist(m, p, s); events.append("Agent 3 (Strategist) wrote the plan using rules (no API key).")
    review = review_plan(recs, s, m, p, call if source == "LLM" else None, events)
    events.append("Agent 4 (Manager) " + ("approved the plan." if review["approved"] else f"rejected the plan. {review['feedback']}"))
    if not review["approved"] and source == "LLM":
        revisions = 1
        try:
            recs = llm_strategist(call, s, m, p, review["feedback"]); events.append("Strategist revised the plan after Manager feedback.")
        except Exception as e:
            events.append(f"Revision failed ({e}); used rule-based plan."); recs = rule_strategist(m, p, s); source = "Rules (fallback)"
        review = review_plan(recs, s, m, p, call if source == "LLM" else None, events)
        events.append("Manager re-reviewed: " + ("approved." if review["approved"] else "still has issues. A human must review."))
    return {"metrics": m, "perf": p, "recs": recs, "review": review, "revisions": revisions, "source": source, "events": events}
