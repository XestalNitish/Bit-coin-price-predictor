# ₿ Bitcoin Close Price Predictor

A Streamlit web app that predicts the **Bitcoin Close Price** using a trained Artificial Neural Network (ANN).

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## 📂 Project Structure

```
├── app.py               # Streamlit application
├── btc_model.keras       # Trained ANN model
├── scaler_X.joblib       # StandardScaler for input features
├── scaler_y.joblib       # StandardScaler for target (Close price)
├── features.json         # Feature names used during training
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## 🧠 Model Details

| Property       | Value                                                      |
|----------------|------------------------------------------------------------|
| Architecture   | Sequential ANN (9 Dense layers)                            |
| Input Features | Open, High, Low, Volume, hour, day, weekday, month, is_weekend |
| Output         | Predicted Close Price (USD)                                |
| Scaling        | StandardScaler (scikit-learn)                              |
| Parameters     | ~48K trainable                                             |

## 🛠️ Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/btc-price-predictor.git
cd btc-price-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **"New app"** → select this repo → set `app.py` as the main file.
4. Click **Deploy** — done!

## 📝 License

MIT
