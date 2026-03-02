from pygooglenews import GoogleNews
import pandas as pd

companies = [
    "TCS", "Infosys", "LTIMindtree", "Tech Mahindra", "Wipro",
    "Accenture", "Cognizant", "Birlasoft", "HDFC",
    "Genpact", "L&T", "Reliance", "Tata Group"
]

gn = GoogleNews(lang='en', country='IN')

ai_trend_data = []

print("Collecting AI news data...")

for company in companies:

    search_term = f"{company} AI OR 'Artificial Intelligence' OR ML"
    search_results = gn.search(search_term, when='1y')

    for entry in search_results['entries']:
        try:
            date = pd.to_datetime(entry['published']).normalize()

            ai_trend_data.append({
                "Company": company,
                "Date": date,
                "AI_Score": 1   # each news = 1 signal
            })

        except:
            continue

# ---------------------------
# Create dataframe
# ---------------------------
df_ai = pd.DataFrame(ai_trend_data)

# ✅ DAILY aggregation instead of MONTHLY
ai_score_daily = (
    df_ai
    .groupby(["Company", "Date"])
    .sum()
    .reset_index()
)

# ---------------------------
# Fill missing days
# ---------------------------
final_data = []

for company in companies:

    temp = ai_score_daily[ai_score_daily["Company"] == company]

    full_dates = pd.date_range(
        start=temp["Date"].min(),
        end=temp["Date"].max(),
        freq="D"
    )

    temp = (
        temp.set_index("Date")
            .reindex(full_dates)
            .fillna(0)
            .rename_axis("Date")
            .reset_index()
    )

    temp["Company"] = company
    final_data.append(temp)

ai_score_daily = pd.concat(final_data)

# ---------------------------
# Smooth trend (important)
# ---------------------------
ai_score_daily["AI_Score"] = (
    ai_score_daily
    .groupby("Company")["AI_Score"]
    .transform(lambda x: x.rolling(7, min_periods=1).mean())
)

# Save
ai_score_daily.to_csv("ai_trend_data.csv", index=False)

print("Daily AI Trend Data Saved Successfully")