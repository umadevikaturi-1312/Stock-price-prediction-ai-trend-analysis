from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly
import joblib
import os
from tensorflow.keras.models import load_model
import json

app = Flask(__name__)

# ---------------------------
# Load Data
# ---------------------------
df = pd.read_csv("stock_features_ready.csv", parse_dates=['Date'])
df.set_index('Date', inplace=True)

# Load AI trend
df_ai = pd.read_csv("ai_trend_data.csv", parse_dates=['Date'])
# ---------------------------
# Load Models & Scalers
# ---------------------------
company_models = {}
scalers_X = joblib.load("scalers_X.pkl")
scalers_y = joblib.load("scalers_y.pkl")
company_features_dict = joblib.load("features.pkl")

for file in os.listdir("models"):
    if file.endswith(".keras"):
        company_name = file.replace(".keras", "")
        company_models[company_name] = load_model(os.path.join("models", file))

companies = [c.replace("_Close_Next", "") for c in company_models.keys()]

DISPLAY_NAMES = {
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "LTIM.NS": "LTIMindtree",
    "TECHM.NS": "Tech Mahindra",
    "WIPRO.NS": "Wipro",
    "ACN": "Accenture",
    "CTSH": "Cognizant",
    "BSOFT.NS": "Birlasoft",
    "HDFCBANK.NS": "HDFC Bank",
    "G": "Genpact",
    "LT.NS": "Larsen & Toubro",
    "RELIANCE.NS": "Reliance Industries",
    "TATASTEEL.NS": "Tata Steel"
}

display_companies = [(c, DISPLAY_NAMES.get(c, c)) for c in companies]

timesteps = 60

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html", companies=display_companies)

@app.route("/plot", methods=["POST"])
def plot_stock():
    try:
        company = request.form.get("company")
        target = company + "_Close_Next"

        if target not in company_models:
            return jsonify({"error": f"Model for {company} not found"})

        model = company_models[target]
        scaler_X = scalers_X[target]
        scaler_y = scalers_y[target]

        company_features = company_features_dict[target]
        X = df[company_features].values

        # Prepare last sequence
        last_seq = X[-timesteps:]
        last_seq_scaled = scaler_X.transform(last_seq).reshape(1, timesteps, len(company_features))

        # Predict next 7 days
        future_preds = []
        for _ in range(7):
            next_pred_scaled = model.predict(last_seq_scaled, verbose=0)
            next_pred = scaler_y.inverse_transform(next_pred_scaled)[0, 0]
            future_preds.append(float(next_pred))

            # Update sequence for next step
            new_features = last_seq_scaled[0, -1, :].copy()
            new_features[-1] = next_pred_scaled[0, 0]
            last_seq_scaled = np.roll(last_seq_scaled, -1, axis=1)
            last_seq_scaled[0, -1, :] = new_features

        next_day_price = future_preds[0]

        # ---------------------------
        # DAILY AI Trend Merge
        # ---------------------------

        df_temp = df.copy().reset_index()

        # Filter AI data for selected company
        symbol_to_ai_name = {
            "TCS.NS": "TCS",
            "INFY.NS": "Infosys",
            "LTIM.NS": "LTIMindtree",
            "TECHM.NS": "Tech Mahindra",
            "WIPRO.NS": "Wipro",
            "ACN": "Accenture",
            "CTSH": "Cognizant",
            "BSOFT.NS": "Birlasoft",
            "HDFCBANK.NS": "HDFC",
            "G": "Genpact",
            "LT.NS": "L&T",
            "RELIANCE.NS": "Reliance",
            "TATASTEEL.NS": "Tata Group"
        }

        company_name_for_ai = symbol_to_ai_name.get(company, company)

        # Select AI data for company
        df_ai_company = df_ai[df_ai['Company'] == company_name_for_ai]

        # Merge using DAILY dates
        df_temp = df_temp.merge(
            df_ai_company[['Date', 'AI_Score']],
            on='Date',
            how='left'
        )

        # Fill missing AI values
        df_temp['AI_Score'] = df_temp['AI_Score'].fillna(0)

        # Get last 30 AI values
        ai_raw = df_temp['AI_Score'].values[-30:]
        ai_series = (
            df_temp['AI_Score']
            .rolling(7, min_periods=1)
            .mean()
        )

        ai_last30 = ai_series.values[-30:]

        # Normalize 0–1
        ai_trend = (
            (ai_last30 - np.min(ai_last30)) /
            (np.max(ai_last30) - np.min(ai_last30) + 1e-8)
        )

        ai_trend = ai_trend.tolist()

        ai_trend = [float(v) for v in ai_trend]        

        # Correlation calculation
        price_column = f"{company}_Close"
        recent_prices = df_temp[price_column].values[-30:]

        correlation = pd.Series(recent_prices).corr(pd.Series(ai_trend))

        if correlation > 0.3:
            ai_effect = f"Positive relationship (Correlation: {correlation:.2f})"
        elif correlation < -0.3:
            ai_effect = f"Negative relationship (Correlation: {correlation:.2f})"
        else:
            ai_effect = f"Weak relationship (Correlation: {correlation:.2f})"


        def calculate_trend_direction(ai_values):
            
            x = np.arange(len(ai_values))
            slope = np.polyfit(x, ai_values, 1)[0]

            if slope > 0:
                return "Increasing AI Adoption"
            elif slope < 0:
                return "Decreasing AI Adoption"
            else:
                return "Stable AI Activity"
    
        trend_status = calculate_trend_direction(ai_trend)

        # Actual prices & dates
        price_column = f"{company}_Close"
        actual_prices = df[price_column].values[-30:]
        actual_prices = [float(p) for p in actual_prices]
        actual_dates = [str(d.date()) for d in df.index[-30:]]

        # Future dates
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
        future_dates = [str(d.date()) for d in future_dates]

        # ---------------------------
        # Create Plotly Figure
        # ---------------------------
        fig = go.Figure()

        # Actual Prices (area fill)
        fig.add_trace(go.Scatter(
            x=actual_dates,
            y=actual_prices,
            mode='lines+markers',
            name='Actual Price',
            fill='tozeroy'
        ))

        # AI Trend
        fig.add_trace(go.Scatter(
            x=actual_dates,
            y=ai_trend,
            mode='lines',
            name='AI Trend',
            yaxis='y2'
        ))

        # Predicted Prices
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_preds,
            mode='lines+markers',
            name='Predicted Price',
            line=dict(dash='dash')
        ))

        fig.update_layout(
        title=f"{DISPLAY_NAMES.get(company, company)} - 7 Day Forecast",
        template='plotly_white',
        hovermode='x unified',
        yaxis=dict(title='Stock Price'),
        yaxis2=dict(
            title='AI Trend Score',
            overlaying='y',
            side='right'
        )
    )

        # Convert figure to JSON
        fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return jsonify({
            "figure": fig_json,
            "next_price": round(next_day_price, 2),
            "ai_effect": ai_effect,
            "trend": trend_status
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)