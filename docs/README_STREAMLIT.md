# 🏆 FIFA World Cup 2026 Predictor - Personal Data Science Project

## 🚀 Quick Start Guide

### Step 1: Install Dependencies

Open your terminal/command prompt in the project directory and run:

```bash
pip install -r requirements.txt
```

This will install:
- Streamlit (web app framework)
- Pandas (data manipulation)
- NumPy (numerical operations)
- Scikit-learn (machine learning)
- Plotly (interactive visualizations)

### Step 2: Run the Streamlit App

```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

---

## 📱 Features

The Streamlit app includes 6 interactive pages:

### 🏠 Home
- Project overview and predicted champion
- Key statistics and technologies used

### 📊 Historical Analysis
- Analysis of World Cup matches from 1930-2022
- Top scoring teams
- Goals distribution and trends

### 🎯 2026 Predictions
- All group stage match predictions
- Win/loss/draw statistics
- Downloadable predictions CSV

### 🏆 Tournament Simulation
- Complete knockout stage simulation
- Round of 16, Quarterfinals, Semifinals, Final
- Predicted champion and final standings

### ⚽ Custom Match Predictor
- Predict any match between two teams
- Interactive team selection
- Real-time predictions with win probability

### 📈 Statistics & Insights
- Detailed team performance analysis
- Historical trends and patterns
- Interactive charts and graphs

---

## 🛠️ Troubleshooting

### Port Already in Use
If port 8501 is already in use, run:
```bash
streamlit run app.py --server.port 8502
```

### Models Not Found
If pre-trained models are missing, the app will automatically train new models using the historical data.

### Data Files Missing
Ensure the following directory structure exists:
```
Fifa-WorldCup-Data-Analysis-1930-2026-main/
├── Data/
│   ├── clean_fifa_worldcup_matches.csv
│   ├── clean_fifa_worldcup_fixture.csv
│   └── fifa_worldcup_historical_data.csv
└── Predictions and Models Folder/
    ├── fifa_worldcup_2026_predictions.csv
    ├── fifa_worldcup_2026_standings_Final_Updated.csv
    ├── predicted_tournament_results.csv
    ├── home_goal_model.pkl
    └── away_goal_model.pkl
```

---

## 🎨 Customization

### Change Theme
You can change the Streamlit theme by creating a `.streamlit/config.toml` file:

```toml
[theme]
primaryColor="#1f77b4"
backgroundColor="#ffffff"
secondaryBackgroundColor="#f0f2f6"
textColor="#262730"
font="sans serif"
```

### Modify Pages
Edit `app.py` to customize the pages, add new features, or modify visualizations.

---

## 📝 Technical Notes

- First load may take a few seconds as data is loaded and cached
- The app uses Streamlit's caching to improve performance
- Predictions are based on Random Forest models trained on historical data
- All data processing and model training is done in real-time

---

## 🎯 Usage Tips

1. **Navigation**: Use the sidebar to switch between different pages
2. **Interactive Charts**: Hover over charts for detailed information
3. **Download Data**: Export predictions as CSV from the predictions page
4. **Custom Predictions**: Try the custom match predictor to see any team matchup

---

## 🛠️ Development

This project demonstrates proficiency in:
- **Data Science**: Historical data analysis and preprocessing
- **Machine Learning**: Predictive modeling with scikit-learn
- **Web Development**: Interactive applications with Streamlit
- **Data Visualization**: Interactive charts with Plotly
- **Data Engineering**: Web scraping and data pipeline creation

Enjoy exploring the FIFA World Cup 2026 predictions! ⚽🏆



