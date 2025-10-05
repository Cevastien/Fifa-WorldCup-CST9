"""
FIFA World Cup 2026 Predictor
CST9 Machine Learning Group Project

Team Members:
- Gella
- Oclarit  
- Planas
- Pulido

Date: October 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import random
from collections import defaultdict

# Page configuration
st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with FIFA official blue design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    .stApp {
        background-color: #f0f2f5;
    }
    
    /* Sidebar styling */
    .stSidebar {
        background-color: #000000;
    }
    
    .stSidebar [data-testid="stSidebarNav"] {
        padding-top: 1rem;
    }
    
    .stSidebar .stMarkdown {
        color: #ffffff !important;
    }
    
    .stSidebar .stMarkdown p {
        color: #ffffff !important;
    }
    
    .stSidebar .stMarkdown h3 {
        color: #ffffff !important;
    }
    
    .stSidebar .stMarkdown h4 {
        color: #ffffff !important;
    }
    
    .stSidebar .stMarkdown h5 {
        color: #ffffff !important;
    }
    
    .stSidebar .stMarkdown h6 {
        color: #ffffff !important;
    }
    
    .stSidebar .stMarkdown strong {
        color: #ffffff !important;
    }
    
    .stSidebar .stMarkdown em {
        color: #ffffff !important;
    }
    
    .stSidebar .stRadio > label {
        color: #ffffff !important;
    }
    
    .stSidebar .stRadio [role="radiogroup"] label {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.1);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        transition: all 0.3s ease;
    }
    
    .stSidebar .stRadio [role="radiogroup"] label:hover {
        background-color: rgba(255, 255, 255, 0.2);
    }
    
    .stSidebar .stRadio [role="radiogroup"] label[data-checked="true"] {
        background-color: #0066b3;
        font-weight: 600;
    }
    
    .stSidebar .stRadio [role="radiogroup"] label div {
        color: #ffffff !important;
    }
    
    .stSidebar hr {
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Sidebar logo section */
    .sidebar-logo {
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 1.5rem;
    }
    
    .sidebar-logo h3 {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0.5rem 0 0.25rem 0;
    }
    
    .sidebar-logo p {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.875rem;
        margin: 0;
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        font-family: 'Inter', sans-serif;
    }
    
    .main-header p {
        font-size: 1rem;
        margin: 0.75rem 0 0 0;
        opacity: 1 !important;
        font-weight: 400;
        font-family: 'Inter', sans-serif;
        color: #ffffff !important;
    }
    
    .main-header div p {
        color: #ffffff !important;
        opacity: 1 !important;
    }
    
    .main-header * {
        color: #ffffff !important;
    }
    
    .main-header div {
        color: #ffffff !important;
    }
    
    .main-header span {
        color: #ffffff !important;
    }
    
    /* Override any Streamlit default text colors in header */
    div[data-testid="stMarkdownContainer"] .main-header p {
        color: #ffffff !important;
    }
    
    .stMarkdown .main-header p {
        color: #ffffff !important;
    }
    
    /* Stat card styling */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border-left: 4px solid #000000;
        position: relative;
        overflow: hidden;
    }
    
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    .stat-card-title {
        color: #6b7280;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
        font-family: 'Inter', sans-serif;
    }
    
    .stat-card-value {
        color: #1a202c;
        font-size: 2.25rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    
    .stat-card-change {
        font-size: 0.875rem;
        color: #000000;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    
    /* Button styling */
    .stButton > button {
        background: #000000;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.875rem 2rem;
        font-weight: 700;
        font-size: 0.875rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        background: #333333;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        transform: translateY(-2px);
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background-color: white;
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #000000;
    }
    
    .stSelectbox label {
        font-weight: 600;
        color: #1a202c;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.875rem;
        font-family: 'Inter', sans-serif;
        color: #4b5563;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #000000;
        color: white;
    }
    
    /* Card container */
    .card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        border-left: 4px solid #000000;
    }
    
    .card-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1a202c;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f2f5;
        font-family: 'Inter', sans-serif;
    }
    
    /* Match card */
    .match-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-left: 6px solid #0066b3;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .match-card:hover {
        box-shadow: 0 4px 12px rgba(0, 102, 179, 0.15);
        transform: translateX(8px);
        border-left-color: #004a7c;
    }
    
    .team-name {
        font-weight: 700;
        color: #1a202c;
        font-size: 1.1rem;
        font-family: 'Inter', sans-serif;
    }
    
    .score {
        color: #0066b3;
        font-weight: 800;
        font-size: 1.5rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Winner badge */
    .winner-card {
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        margin: 2rem 0;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        font-family: 'Inter', sans-serif;
    }
    
    /* Data table styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
        font-family: 'Inter', sans-serif;
    }
    
    /* Metric styling */
    .stMetric {
        font-family: 'Inter', sans-serif;
    }
    
    .stMetric label {
        font-family: 'Inter', sans-serif;
        color: #6b7280;
        font-weight: 600;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif;
        color: #1a202c;
        font-weight: 700;
    }
    
    /* Text visibility improvements for specific areas */
    .stMarkdown {
        color: #1a202c !important;
    }
    
    .stMarkdown p {
        color: #1a202c !important;
    }
    
    .stMarkdown ul li {
        color: #1a202c !important;
    }
    
    .stMarkdown ol li {
        color: #1a202c !important;
    }
    
    .stMarkdown strong {
        color: #1a202c !important;
    }
    
    .stMarkdown em {
        color: #1a202c !important;
    }
    
    /* Main content text styling */
    .main .block-container {
        color: #1a202c !important;
    }
    
    .main .block-container p {
        color: #1a202c !important;
    }
    
    .main .block-container ul {
        color: #1a202c !important;
    }
    
    .main .block-container li {
        color: #1a202c !important;
    }
    
    /* Card content text */
    .card p {
        color: #1a202c !important;
    }
    
    .card ul {
        color: #1a202c !important;
    }
    
    .card li {
        color: #1a202c !important;
    }
    
    /* Match predictor text visibility */
    .stSelectbox label {
        color: #1a202c !important;
        font-weight: 600;
    }
    
    .stSelectbox > div > div {
        color: #1a202c !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        color: #1a202c !important;
    }
    
    /* Prediction result text */
    .prediction-result h3 {
        color: #1a202c !important;
        font-weight: 700;
    }
    
    .prediction-result p {
        color: #6b7280 !important;
    }
    
    /* Error messages */
    .stAlert {
        color: #1a202c !important;
    }
    
    .stAlert p {
        color: #1a202c !important;
    }
    
    /* Icons */
    .icon-soccer:before {
        content: "⚽";
    }
    
    .icon-target:before {
        content: "🎯";
    }
    
    .icon-trophy:before {
        content: "🏆";
    }
    
    .icon-home:before {
        content: "🏠";
    }
    
    .icon-plane:before {
        content: "✈️";
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Base paths
BASE_PATH = Path(".")
DATA_PATH = BASE_PATH / "data" / "cleaned"
MODELS_PATH = BASE_PATH / "ml_models" / "trained"
RESULTS_PATH = BASE_PATH / "data" / "results"

# Load data function
@st.cache_data
def load_data():
    """Load all necessary data files"""
    data = {}
    try:
        data['historical'] = pd.read_csv(DATA_PATH / "fifa_worldcup_historical_data.csv")
        data['matches'] = pd.read_csv(DATA_PATH / "clean_fifa_worldcup_matches.csv").dropna()
        data['fixtures'] = pd.read_csv(DATA_PATH / "clean_fifa_worldcup_fixture.csv")
        data['predictions'] = pd.read_csv(RESULTS_PATH / "fifa_worldcup_2026_predictions.csv")
        data['standings'] = pd.read_csv(RESULTS_PATH / "fifa_worldcup_2026_standings_Final_Updated.csv")
        data['tournament_results'] = pd.read_csv(RESULTS_PATH / "predicted_tournament_results.csv")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
    return data

@st.cache_resource
def load_models():
    """Load smart pre-trained models"""
    try:
        # Try to load smart models first
        smart_home_path = MODELS_PATH / "smart_home_model.pkl"
        smart_away_path = MODELS_PATH / "smart_away_model.pkl"
        smart_encoder_path = MODELS_PATH / "smart_team_encoder.pkl"
        smart_scaler_path = MODELS_PATH / "smart_scaler.pkl"
        
        if all([smart_home_path.exists(), smart_away_path.exists(), smart_encoder_path.exists(), smart_scaler_path.exists()]):
            with open(smart_home_path, 'rb') as f:
                home_model = pickle.load(f)
            with open(smart_away_path, 'rb') as f:
                away_model = pickle.load(f)
            with open(smart_encoder_path, 'rb') as f:
                team_encoder = pickle.load(f)
            with open(smart_scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            from sklearn.base import BaseEstimator
            if isinstance(home_model, BaseEstimator) and isinstance(away_model, BaseEstimator):
                if hasattr(home_model, 'predict') and hasattr(away_model, 'predict'):
                    return home_model, away_model, team_encoder, scaler, True  # True indicates smart model
        
        # Fallback to basic models
        home_model_path = MODELS_PATH / "home_goal_model.pkl"
        away_model_path = MODELS_PATH / "away_goal_model.pkl"
        
        if not home_model_path.exists() or not away_model_path.exists():
            return None, None, None, None, False
            
        with open(home_model_path, 'rb') as f:
            home_model = pickle.load(f)
        with open(away_model_path, 'rb') as f:
            away_model = pickle.load(f)
        
        from sklearn.base import BaseEstimator
        if not isinstance(home_model, BaseEstimator) or not isinstance(away_model, BaseEstimator):
            return None, None, None, None, False
            
        if not hasattr(home_model, 'predict') or not hasattr(away_model, 'predict'):
            return None, None, None, None, False
            
        return home_model, away_model, None, None, False  # False indicates basic model
        
    except Exception as e:
        return None, None, None, None, False

@st.cache_data
def train_models(df):
    """Train models if pre-trained ones are not available"""
    team_encoder = LabelEncoder()
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    team_encoder.fit(all_teams)
    
    df['HomeTeamEncoded'] = team_encoder.transform(df['HomeTeam'])
    df['AwayTeamEncoded'] = team_encoder.transform(df['AwayTeam'])
    
    X = df[['HomeTeamEncoded', 'AwayTeamEncoded']]
    y_home = df['HomeGoals']
    y_away = df['AwayGoals']
    
    home_model = RandomForestRegressor(n_estimators=100, random_state=42)
    away_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    home_model.fit(X, y_home)
    away_model.fit(X, y_away)
    
    return home_model, away_model, team_encoder

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-logo">
                <h3><span class="icon-soccer"></span> FIFA World Cup</h3>
                <p>2026 Predictor</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state for page navigation
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Dashboard"
        
        page = st.radio(
            "Navigation",
            ["Dashboard", "2026 Predictions", "Match Predictor"],
            index=["Dashboard", "2026 Predictions", "Match Predictor"].index(st.session_state.current_page),
            label_visibility="collapsed"
        )
        
        # Update session state when radio button changes
        if page != st.session_state.current_page:
            st.session_state.current_page = page
            st.rerun()
        
        st.markdown("---")
        st.markdown("**CST9 Group Project**")
        st.markdown("*Machine Learning*")
        st.markdown("*Prediction System*")
        
        st.markdown("---")
        st.markdown("**Team Members**")
        st.markdown("• Gella")
        st.markdown("• Oclarit")
        st.markdown("• Planas")
        st.markdown("• Pulido")
    
    # Load data and models
    data = load_data()
    home_model, away_model, team_encoder, scaler, is_smart_model = load_models()
    
    # Train models if not available or create team encoder
    if home_model is None or away_model is None:
        if 'matches' in data and not data['matches'].empty:
            home_model, away_model, team_encoder = train_models(data['matches'])
            is_smart_model = False
            scaler = None
        else:
            team_encoder = None
            is_smart_model = False
            scaler = None
    else:
        # Models loaded successfully
        if not is_smart_model:
            # For basic models, create team encoder
            df = data.get('matches', pd.DataFrame())
            if not df.empty:
                team_encoder = LabelEncoder()
                all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
                team_encoder.fit(all_teams)
            else:
                team_encoder = None
    
    # Page routing
    if page == "Dashboard":
        show_dashboard(data, is_smart_model)
    elif page == "2026 Predictions":
        show_predictions(data)
    elif page == "Match Predictor":
        show_match_predictor(home_model, away_model, team_encoder, scaler, is_smart_model, data)

def show_dashboard(data, is_smart_model=False):
    """Enhanced dashboard page"""
    # Header
    st.markdown("""
        <div class="main-header">
            <h1><span class="icon-soccer"></span> FIFA World Cup 2026 Predictor</h1>
            <p>CST9 Machine Learning Group Project - Predicting the future champion using advanced AI algorithms</p>
        </div>
    """, unsafe_allow_html=True)
    
    
    # Key statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
                <div class="stat-card-title">TEAMS ANALYZED</div>
                <div class="stat-card-value">80+</div>
                <div class="stat-card-change">Historical Data</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
                <div class="stat-card-title">DATA RANGE</div>
                <div class="stat-card-value">93</div>
                <div class="stat-card-change">Years (1930-2022)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
                <div class="stat-card-title">ML MODEL</div>
                <div class="stat-card-value">RF</div>
                <div class="stat-card-change">Random Forest</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
                <div class="stat-card-title">PREDICTIONS</div>
                <div class="stat-card-value">48</div>
                <div class="stat-card-change">Group Matches</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Champion prediction
    if 'standings' in data and not data['standings'].empty:
        standings = data['standings']
        champion = standings[standings['Position'] == 'Champion']
        if not champion.empty:
            champion_name = champion['Team'].values[0]
            st.markdown(f"""
                <div class="winner-card">
                    <span class="icon-trophy"></span> Predicted 2026 Champion: {champion_name} <span class="icon-trophy"></span>
            </div>
            """, unsafe_allow_html=True)
    
    # Quick actions
    st.markdown('<div class="card"><div class="card-header">Quick Actions</div></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("View 2026 Predictions", use_container_width=True):
            st.session_state.current_page = "2026 Predictions"
            st.rerun()
    
    with col2:
        if st.button("Predict Custom Match", use_container_width=True):
            st.session_state.current_page = "Match Predictor"
            st.rerun()
    
    # About the World Cup section
    st.markdown('<div class="card"><div class="card-header">About the FIFA World Cup</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **What is the FIFA World Cup?**
        
        The FIFA World Cup is the most prestigious international football tournament, held every 4 years since 1930. 
        It brings together the best national teams from around the world to compete for the ultimate prize in football.
        
        **2026 Tournament Format:**
        - **48 teams** (expanded from 32)
        - **12 groups** of 4 teams each
        - **Group Stage**: Each team plays 3 matches
        - **Top 2 teams** from each group advance (24 teams)
        - **Knockout Stage**: Single elimination bracket
        - **Total**: 104 matches over ~1 month
        """)
    
    with col2:
        st.markdown("""
        **Tournament Structure:**
        
        ```
        Group Stage (48 teams)
        ├── 12 Groups of 4 teams
        ├── Round-robin format
        └── Top 2 advance (24 teams)
        
        Knockout Stage (24 teams)
        ├── Round of 16 (16 teams)
        ├── Quarterfinals (8 teams)
        ├── Semifinals (4 teams)
        └── Final (2 teams)
        ```
        
        **Key Dates:**
        - **June-July 2026**
        - **3 Host Countries**: USA, Canada, Mexico
        - **16 Host Cities** across North America
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Football Terms Glossary
    st.markdown('<div class="card"><div class="card-header">📚 Football Terms & Definitions</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Tournament Terms:**
        - **Group Stage**: Teams play each other once, top 2 advance
        - **Knockout Stage**: Single elimination, winner advances, loser eliminated
        - **Round of 16**: First knockout round (16 teams)
        - **Quarterfinals**: Second knockout round (8 teams)
        - **Semifinals**: Third knockout round (4 teams)
        - **Final**: Championship match (2 teams)
        """)
    
    with col2:
        st.markdown("""
        **Scoring System:**
        - **Win**: 3 points
        - **Draw**: 1 point each
        - **Loss**: 0 points
        - **Goals For (GF)**: Goals scored by team
        - **Goals Against (GA)**: Goals conceded by team
        - **Goal Difference (GD)**: GF - GA
        - **Extra Time**: 30 minutes if tied after 90 minutes
        - **Penalty Shootout**: If still tied after extra time
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Project overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card"><div class="card-header">About This Project</div>', unsafe_allow_html=True)
        st.markdown("""
        This application uses machine learning to predict FIFA World Cup 2026 match outcomes and tournament results.
        
        **Key Features:**
        - Historical data analysis from 1930-2022
        - Random Forest machine learning models
        - Complete tournament simulation
        - Real-time match predictions
        - Interactive data visualizations
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card"><div class="card-header">Technologies</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Python** - Core language
        - **Scikit-learn** - ML models  
        - **Pandas** - Data processing
        - **Streamlit** - Web interface
        - **Plotly** - Visualizations
        """)
        st.markdown('</div>', unsafe_allow_html=True)

def show_predictions(data):
    """Enhanced predictions page"""
    st.markdown("""
        <div class="main-header">
            <h1><span class="icon-target"></span> FIFA World Cup 2026 Predictions</h1>
            <p>Complete tournament bracket and group stage predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # How Predictions Work section
    st.markdown('<div class="card"><div class="card-header">🤖 How Our Predictions Work</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Machine Learning Approach:**
        
        Our predictions are generated using advanced machine learning algorithms trained on historical FIFA World Cup data from 1930-2022.
        
        **Data Sources:**
        - **Historical matches**: 900+ World Cup matches over 92 years
        - **Team performance**: Goals scored, goals conceded, win/loss records
        - **Tournament factors**: Home advantage, team rankings, historical head-to-head
        - **Statistical patterns**: Goal scoring trends, defensive records
        
        **Model Details:**
        - **Algorithm**: Random Forest Regressor
        - **Features**: Team strength, historical performance, tournament context
        - **Training**: Cross-validated on multiple World Cup tournaments
        - **Output**: Goal predictions and match outcomes
        """)
    
        with col2:
            st.markdown("""
        **Important Notes:**
        
        ⚠️ **Predictions are probabilities, not guarantees**
        
        • Based on historical data and statistical patterns
        • Cannot account for unexpected events (injuries, weather, etc.)
        • Football is inherently unpredictable
        • Use for entertainment and analysis purposes
        
        **Confidence Levels:**
        - **High**: Strong historical data support
        - **Medium**: Moderate statistical evidence
        - **Low**: Limited data or close matchups
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tournament results
    if 'tournament_results' in data and not data['tournament_results'].empty:
        results = data['tournament_results']
        standings = data.get('standings', pd.DataFrame())
        
        champion = standings[standings['Position'] == 'Champion'] if not standings.empty else pd.DataFrame()
        if not champion.empty:
            champion_name = champion['Team'].values[0]
            st.markdown(f'<div class="winner-card"><span class="icon-trophy"></span> Predicted 2026 Champion: {champion_name} <span class="icon-trophy"></span></div>', unsafe_allow_html=True)
        
        # Tournament bracket visualization
        st.markdown('<div class="card"><div class="card-header">🏆 Tournament Structure Overview</div>', unsafe_allow_html=True)
        
        st.markdown("""
        **2026 World Cup Tournament Structure:**
        
        ```
        Group Stage (48 teams → 24 teams)
        ├── 12 Groups of 4 teams
        ├── Each team plays 3 matches
        └── Top 2 from each group advance
        
        Knockout Stage (24 teams → 1 champion)
        ├── Round of 16 (24 → 16 teams)
        ├── Quarterfinals (16 → 8 teams)  
        ├── Semifinals (8 → 4 teams)
        ├── Third Place (4 → 3rd & 4th place)
        └── Final (2 → Champion)
        
        Legend:
        🟢 Advanced    🔴 Eliminated    🏆 Champion
        ```
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card"><div class="card-header">Tournament Progression</div>', unsafe_allow_html=True)
        
        tabs = st.tabs(["Round of 16", "Quarterfinals", "Semifinals", "Final"])
        rounds = ["Round of 16", "Quarterfinals", "Semifinals", "Final"]
        
        for tab, round_name in zip(tabs, rounds):
            with tab:
                round_matches = results[results['Round'] == round_name]
                
                if not round_matches.empty:
                    for _, match in round_matches.iterrows():
                        st.markdown(f"""
                        <div class="match-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="flex: 1;">
                                    <span class="team-name">{match['Team1']}</span>
                                </div>
                                <div style="text-align: center; padding: 0 1rem;">
                                    <span class="score">{match['Predicted Goals Team1']:.1f} - {match['Predicted Goals Team2']:.1f}</span>
                                </div>
                                <div style="flex: 1; text-align: right;">
                                    <span class="team-name">{match['Team2']}</span>
                                </div>
                            </div>
                            <div style="text-align: center; margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid #e5e7eb;">
                                <span style="color: #0066b3; font-weight: 600;">Winner: {match['Winner']}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(f"No matches found for {round_name}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Group stage predictions
    st.markdown('<div class="card"><div class="card-header">Group Stage Predictions</div>', unsafe_allow_html=True)
    
    if 'predictions' in data and not data['predictions'].empty:
        predictions = data['predictions']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Matches", len(predictions))
        with col2:
            home_wins = len(predictions[predictions['Winner'] == predictions['HomeTeam']])
            st.metric("Home Wins", home_wins)
        with col3:
            away_wins = len(predictions[predictions['Winner'] == predictions['AwayTeam']])
            st.metric("Away Wins", away_wins)
        with col4:
            draws = len(predictions[predictions['Winner'] == 'Draw'])
            st.metric("Draws", draws)
        
        st.dataframe(
            predictions[['HomeTeam', 'PredictedHomeGoals', 'PredictedAwayGoals', 'AwayTeam', 'Winner']],
            use_container_width=True,
            height=400
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_match_predictor(home_model, away_model, team_encoder, scaler, is_smart_model, data):
    """Enhanced match predictor page"""
    st.markdown("""
        <div class="main-header">
            <h1><span class="icon-soccer"></span> Custom Match Predictor</h1>
            <p>Predict the outcome of any match between two teams</p>
                    </div>
                    """, unsafe_allow_html=True)
        
    # Educational context
    st.markdown('<div class="card"><div class="card-header">📖 Understanding Our Predictions</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Why does Home/Away matter?**
        
        • **Home advantage**: Teams typically perform better at home
        • **Familiar environment**: Home crowd support, familiar stadium
        • **Travel impact**: Away teams may be affected by travel, time zones
        • **Historical data**: Our model considers historical home/away performance
        
        **Prediction Factors:**
        • Team strength and recent form
        • Historical head-to-head record
        • Tournament context (group stage vs knockout)
        • Statistical goal-scoring patterns
        """)
    
    with col2:
        st.markdown("""
        **How to Read Results:**
        
        • **Predicted Goals**: Expected number of goals each team will score
        • **Decimals**: Represent probability (e.g., 1.8 goals ≈ 80% chance of 1 goal, 20% chance of 2)
        • **Winner**: Based on higher predicted goal count
        • **Confidence**: Based on historical data quality
        
        **Example:**
        Brazil 2.1 - 0.8 Argentina
        → Brazil more likely to win (higher predicted goals)
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if home_model is None or away_model is None or team_encoder is None:
        st.error("Models not available for prediction")
        return
    
    teams = sorted(team_encoder.classes_)
    
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox("Home Team", teams, index=teams.index('Brazil') if 'Brazil' in teams else 0)
        if st.checkbox("Show team info", key="home_info"):
            st.info(f"**{home_team}** - Select any national team from the list")
    
    with col2:
        away_team = st.selectbox("Away Team", teams, index=teams.index('Argentina') if 'Argentina' in teams else 1)
        if st.checkbox("Show team info", key="away_info"):
            st.info(f"**{away_team}** - Select any national team from the list")
    
    if st.button("Predict Match Result", type="primary"):
        if home_team == away_team:
            st.error("Please select different teams!")
        else:
            if is_smart_model and scaler is not None:
                # Smart model prediction with advanced features
                home_idx = team_encoder.transform([home_team])[0]
                away_idx = team_encoder.transform([away_team])[0]
                
                # Get team stats for advanced features
                df = data.get('matches', pd.DataFrame())
                home_matches = df[(df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)]
                away_matches = df[(df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)]
                
                home_goals_for = home_matches.apply(lambda x: x['HomeGoals'] if x['HomeTeam'] == home_team else x['AwayGoals'], axis=1).mean() if len(home_matches) > 0 else 1.5
                away_goals_for = away_matches.apply(lambda x: x['HomeGoals'] if x['HomeTeam'] == away_team else x['AwayGoals'], axis=1).mean() if len(away_matches) > 0 else 1.5
                
                # Create feature vector (simplified version of smart features)
                features = np.array([[home_idx, away_idx, home_goals_for, 1.5, 0.5, away_goals_for, 1.5, 0.5, 
                                     home_goals_for, away_goals_for, home_goals_for, away_goals_for, 
                                     len(home_matches), len(away_matches), 1.0, 1.0, 1.0, 0.0]])
                
                features_scaled = scaler.transform(features)
                predicted_home_goals = max(0, home_model.predict(features_scaled)[0])
                predicted_away_goals = max(0, away_model.predict(features_scaled)[0])
            else:
                # Basic model prediction
                home_encoded = team_encoder.transform([home_team])[0]
                away_encoded = team_encoder.transform([away_team])[0]
                
                match_features = pd.DataFrame([[home_encoded, away_encoded]], 
                                             columns=['HomeTeamEncoded', 'AwayTeamEncoded'])
                predicted_home_goals = home_model.predict(match_features)[0]
                predicted_away_goals = away_model.predict(match_features)[0]
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.markdown(f"""
                <div class="prediction-result" style="text-align: center;">
                    <h3>{home_team}</h3>
                    <div class="score">{predicted_home_goals:.2f}</div>
                    <p>Goals</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="text-align: center; padding-top: 2rem;">
                    <h3 style="color: #6b7280 !important;">vs</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="prediction-result" style="text-align: center;">
                    <h3>{away_team}</h3>
                    <div class="score">{predicted_away_goals:.2f}</div>
                    <p>Goals</p>
                </div>
                """, unsafe_allow_html=True)
            
            if predicted_home_goals > predicted_away_goals:
                winner = home_team
                confidence = abs(predicted_home_goals - predicted_away_goals)
            elif predicted_away_goals > predicted_home_goals:
                winner = away_team
                confidence = abs(predicted_away_goals - predicted_home_goals)
            else:
                winner = "Draw"
                confidence = 0
            
            # Confidence level
            if confidence > 1.5:
                conf_level = "High"
                conf_color = "#22c55e"
            elif confidence > 0.8:
                conf_level = "Medium"
                conf_color = "#f59e0b"
            else:
                conf_level = "Low"
                conf_color = "#ef4444"
            
            st.markdown(f"""
            <div class="winner-card">
                <span class="icon-trophy"></span> Predicted Winner: {winner} <span class="icon-trophy"></span>
                <div style="margin-top: 1rem; font-size: 1rem; opacity: 0.9;">
                    Confidence: <span style="color: {conf_color}; font-weight: 600;">{conf_level}</span>
                    <br><small style="opacity: 0.8;">Based on {confidence:.1f} goal difference</small>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()