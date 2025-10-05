# FIFA World Cup 2026 Predictor

## 🏆 Personal Data Science Project

A **Machine Learning** project that predicts the **FIFA World Cup 2026** results, from **group stage to the final champion**! ⚽🔥  

This project demonstrates my skills in **data scraping, data cleaning, predictive modeling, and knockout round simulation** to forecast match results using historical FIFA World Cup data (1930-2022).

---

## 🚀 Project Overview

🔹 **Data Collection**: Extracted historical World Cup match results from **Wikipedia (1930-2022)**  
🔹 **Data Cleaning & Structuring**: Processed and formatted data into a structured dataset  
🔹 **Fixture Generation**: Created possible **2026 group-stage fixtures**  
🔹 **Match Predictions**: Trained ML models to **predict match outcomes (home & away goals)**  
🔹 **Knockout Simulation**: Simulated each stage, determining the **winner of the tournament**  
🔹 **Interactive Web App**: Built a Streamlit application for visualization and predictions  
🔹 **Data Export**: Outputs match predictions & tournament standings in CSV format  

---

## ⚡ Technologies & Skills Used

✅ **Python** – Core programming language  
✅ **Pandas** – Data cleaning, structuring & manipulation  
✅ **BeautifulSoup** – Web scraping from Wikipedia  
✅ **Scikit-learn** – Machine learning models (Random Forest) for match predictions  
✅ **NumPy** – Mathematical operations & simulations  
✅ **Streamlit** – Interactive web application  
✅ **Plotly** – Interactive data visualization  
✅ **Jupyter Notebook** – Development & testing environment  

---
## 🎯 Features & Workflow  

### 1️⃣ Data Collection & Preprocessing  
- Extracts **FIFA World Cup match history (1930-2022)** from Wikipedia  
- Cleans and structures data into a usable format  

### 2️⃣ Fixture Generation  
- Generates **possible 2026 group-stage fixtures**  
- Saves to `fixtures_2026.csv`  

### 3️⃣ Machine Learning Model Training  
- **Encodes team strengths**  
- Trains **ML models** to predict **home & away goals**  
- Uses **Random Forest Regressor** for score prediction  

### 4️⃣ Tournament Simulation  
- Predicts **group stage results**  
- Simulates **knockout rounds** (Round of 16, Quarterfinals, Semifinals, Final)  
- Determines **FIFA World Cup 2026 Champion** 🏆  

### 5️⃣ Interactive Web Application  
- Built with **Streamlit** for interactive exploration  
- Features historical analysis, predictions, and custom match predictor  
- Real-time visualization with **Plotly** charts  

### 6️⃣ Data Export  
- Saves all results, including **match predictions & tournament standings**  

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
1. Clone or download this project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## 📊 Project Structure

```
├── app.py                              # Streamlit web application
├── requirements.txt                     # Python dependencies
├── run_streamlit.bat                   # Windows batch file to run app
├── README_STREAMLIT.md                 # Streamlit app documentation
├── Fifa-WorldCup-Data-Analysis-1930-2026-main/
│   ├── Data/                           # Historical match data
│   ├── Predictions and Models Folder/  # ML models and predictions
│   └── *.ipynb                         # Jupyter notebooks for analysis
```

---

## 🎯 Key Features

- **Historical Analysis**: Explore World Cup data from 1930-2022
- **2026 Predictions**: View all group stage match predictions
- **Tournament Simulation**: Complete knockout stage simulation
- **Custom Predictor**: Predict any match between two teams
- **Interactive Visualizations**: Charts and graphs for data exploration
- **Export Functionality**: Download predictions as CSV files

---

## 📈 Future Enhancements

🔹 Improve ML accuracy with advanced models  
🔹 Add expected goals (xG) analysis  
🔹 Optimize knockout stage simulation  
🔹 Add more historical data sources  
🔹 Implement team ranking algorithms  

---

*This project showcases my skills in data science, machine learning, and web application development.*
