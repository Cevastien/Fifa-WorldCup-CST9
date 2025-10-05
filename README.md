
## Project Overview

This project predicts the FIFA World Cup 2026 results using machine learning techniques. We utilize historical World Cup data from 1930-2022 to train models that can forecast match outcomes and tournament progression.

## Features

- **Match Prediction**: Predict individual match outcomes
- **Tournament Simulation**: Complete bracket simulation from group stage to final
- **Interactive Web App**: User-friendly Streamlit interface
- **Historical Analysis**: Insights from 90+ years of World Cup data

## Project Structure

```
FIFA-WorldCup-Predictor-CST9/
├── data/
│   ├── raw/           # Original scraped data
│   ├── cleaned/       # Processed training data
│   └── results/       # Prediction outputs
├── ml_models/
│   ├── trained/       # Trained model files
│   └── encoders/      # Label encoders
├── notebooks/
│   ├── 01_web_scraping.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_tournament_simulation.ipynb
├── streamlit_app/
│   └── app.py         # Main application
├── docs/
│   ├── README.md
│   ├── README_STREAMLIT.md
│   └── PROJECT_REPORT.md
├── requirements.txt
└── .gitignore
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd FIFA-WorldCup-Predictor-CST9
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run streamlit_app/app.py
```

## Usage

1. **Home Page**: Overview and quick actions
2. **2026 Predictions**: View complete tournament predictions
3. **Custom Match Predictor**: Predict any team matchup

## Technologies Used

- **Python**: Core programming language
- **Scikit-learn**: Machine learning models
- **Pandas & NumPy**: Data manipulation
- **Streamlit**: Web application framework
- **Plotly**: Data visualization
- **BeautifulSoup**: Web scraping

## Methodology

1. **Data Collection**: Historical FIFA World Cup data (1930-2022)
2. **Data Cleaning**: Standardize team names, handle missing values
3. **Model Training**: Random Forest Regressor for goal prediction
4. **Tournament Simulation**: Iterative prediction through tournament stages
5. **Web Application**: Interactive interface for predictions

## Results

The application provides:
- Predicted tournament champion
- Complete bracket simulation
- Match-by-match predictions
- Interactive visualizations
- Export functionality

## Team Contributions

See `docs/PROJECT_REPORT.md` for detailed individual contributions and responsibilities.

## Future Improvements

- Enhanced player-level statistics
- Advanced neural network models
- Real-time match data integration
- Improved user interface

## License

This project is for educational purposes as part of CST9 group coursework.

---

**Note**: This is a collaborative group project. See `docs/PROJECT_REPORT.md` for individual team member contributions.
