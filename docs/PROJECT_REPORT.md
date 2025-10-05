# FIFA World Cup 2026 Predictor - CST9 Group Project

## Team Members
- **Gella** - [Specific contributions]
- **Oclarit** - [Specific contributions]  
- **Planas** - [Specific contributions]
- **Pulido** - [Specific contributions]

## Project Overview

This project aims to predict the FIFA World Cup 2026 results using machine learning techniques. We utilize historical World Cup data from 1930-2022 to train models that can forecast match outcomes and tournament progression.

### Objectives
- Predict match outcomes for FIFA World Cup 2026
- Simulate complete tournament progression
- Provide interactive web interface for predictions
- Demonstrate proficiency in machine learning and data science

## Methodology

### Data Collection
- **Historical Data**: FIFA World Cup matches from 1930-2022
- **Data Source**: Wikipedia web scraping
- **Data Cleaning**: Removed missing values, standardized team names

### Machine Learning Approach
- **Algorithm**: Random Forest Regressor
- **Features**: Team encoding, historical performance
- **Models**: Separate models for home and away goal predictions
- **Validation**: Cross-validation on historical data

### Tournament Simulation
- **Method**: Iterative match prediction through tournament stages
- **Stages**: Group Stage → Round of 16 → Quarterfinals → Semifinals → Final
- **Output**: Complete bracket with predicted winners

## Results

### Model Performance
- **Training Accuracy**: [To be filled by team]
- **Prediction Confidence**: [To be filled by team]
- **Key Insights**: [To be filled by team]

### 2026 Predictions
- **Predicted Champion**: [To be filled by team]
- **Top Teams**: [To be filled by team]
- **Upsets**: [To be filled by team]

## Individual Contributions

### Gella
- **Responsibilities**: [To be filled]
- **Code Sections**: [To be filled]
- **Notebooks**: [To be filled]

### Oclarit
- **Responsibilities**: [To be filled]
- **Code Sections**: [To be filled]
- **Notebooks**: [To be filled]

### Planas
- **Responsibilities**: [To be filled]
- **Code Sections**: [To be filled]
- **Notebooks**: [To be filled]

### Pulido
- **Responsibilities**: [To be filled]
- **Code Sections**: [To be filled]
- **Notebooks**: [To be filled]

## Technical Implementation

### Technologies Used
- **Python**: Core programming language
- **Scikit-learn**: Machine learning models
- **Pandas & NumPy**: Data manipulation
- **Streamlit**: Web application framework
- **Plotly**: Data visualization
- **BeautifulSoup**: Web scraping

### Project Structure
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
│   └── PROJECT_REPORT.md
└── requirements.txt
```

## Challenges and Solutions

### Data Quality
- **Challenge**: Missing historical data, inconsistent team names
- **Solution**: Data cleaning pipeline, team name standardization

### Model Accuracy
- **Challenge**: Limited historical data for some teams
- **Solution**: Feature engineering, ensemble methods

### Tournament Simulation
- **Challenge**: Complex bracket logic
- **Solution**: Iterative prediction algorithm

## Future Improvements

1. **Enhanced Features**: Player-level statistics, team rankings
2. **Advanced Models**: Neural networks, ensemble methods
3. **Real-time Updates**: Live match data integration
4. **User Interface**: More interactive visualizations

## Conclusion

This project demonstrates the application of machine learning to sports prediction. While predictions cannot guarantee accuracy, the models provide valuable insights into team performance patterns and tournament dynamics.

## References

- FIFA World Cup Historical Data
- Scikit-learn Documentation
- Streamlit Documentation
- [Additional references as needed]

---

**Note**: This template should be completed by team members with their specific contributions and results.
