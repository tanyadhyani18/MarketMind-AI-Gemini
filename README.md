# MarketMind AI
Agentic marketing intelligence app (Streamlit) for an MBA Marketing + Big Data Analytics project.

## Agents
1. **Data Analyst** – portfolio metrics (spend, revenue, conversions, ROAS, CTR, conversion rate, CPA). Pure Python.
2. **Campaign Performance Analyst** – best/weakest ROAS, funnel leaks (high CTR, low conversion). Pure Python.
3. **Marketing Strategist** – concrete actions with budget shift, evidence, KPI, expected impact. Uses an **LLM** when an API key is given, otherwise transparent rules.
4. **Manager (reviewer)** – checks the plan (real campaigns named, KPIs present, budget shift ≤ 20% of spend, **every number traceable to the data**, validation step). Rejects and requests ONE revision if it fails; a human must still approve.

The LLM never calculates numbers. It only receives numbers computed by agents 1-2, and the Manager verifies them.

## CSV format
Required: `Campaign, Channel, Impressions, Clicks, Conversions, Spend, Revenue`. Optional: `Date` (enables the monthly ROAS trend).

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## AI mode (free)
Get a free key at aistudio.google.com. Either paste it in the sidebar, or on Streamlit Cloud open **Manage app → Settings → Secrets** and add:
```
LLM_API_KEY = "your-key"
```
(Optional: `LLM_BASE_URL`, `MODEL_NAME`.) Never commit keys to GitHub.

## Limitations
ROAS is not profit. Sample data is illustrative. Keyword sentiment is approximate. Recommendations support decisions; they do not make them.
