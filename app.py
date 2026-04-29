import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# --- INITIALIZATION ---
st.set_page_config(page_title="IPL Pro Predictor v4", page_icon="🏏", layout="wide")

@st.cache_resource
def load_ipl_resources():
    # We load the dictionary containing the model AND the trained columns
    with open('ipl_model_v4.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['columns']

# Load model and the exact column list used during training
model, model_columns = load_ipl_resources()

# --- SIDEBAR MATCH CONTROL ---
st.sidebar.header("🏟️ Match Setup")
teams = [
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans',
    'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians',
    'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
]
cities = [
    'Mumbai', 'Chennai', 'Delhi', 'Kolkata', 'Hyderabad', 
    'Bangalore', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Chandigarh'
]

batting_team = st.sidebar.selectbox("Batting Team (Chasing)", sorted(teams))
bowling_team = st.sidebar.selectbox("Bowling Team", sorted([t for t in teams if t != batting_team]))
city = st.sidebar.selectbox("Host City", sorted(cities))
target = st.sidebar.number_input("Target Score", 1, 300, 180)

# --- MAIN DASHBOARD ---
st.title("🏏 IPL Win Probability Predictor")
st.markdown("#### *Powered by Ball-by-Ball Historical Data (2008-2025)*")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Match Situation")
    c1, c2, c3 = st.columns(3)
    with c1:
        runs_total = st.number_input("Current Score", 0, target + 10, 100)
    with c2:
        wickets_lost = st.number_input("Wickets Down", 0, 10, 3)
    with c3:
        over = st.number_input("Overs Completed", 0, 19, 10)

    # Calculation Logic
    runs_required = target - runs_total
    balls_remaining = 120 - (over * 6)
    wickets_remaining = 10 - wickets_lost
    rrr = round(runs_required / (balls_remaining / 6), 2) if balls_remaining > 0 else 99
    crr = round(runs_total / (over if over > 0 else 1), 2)

with col2:
    st.subheader("📈 Live Metrics")
    st.metric("Runs Needed", f"{runs_required} off {balls_remaining} balls")
    st.metric("Required Rate", f"{rrr}")


def get_prediction():
    # 1. Immediate Mathematical Overrides
    if runs_required <= 0: return 100
    if wickets_remaining <= 0 or (balls_remaining <= 0 and runs_required > 0): return 0.0
    
    if balls_remaining > 0:
        max_theoretical_runs = balls_remaining * 6 
        if runs_required > max_theoretical_runs:
            return 0.0

    # 2. Match the Training Columns
    input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
    input_df['runs_left'] = runs_required
    input_df['balls_left'] = balls_remaining
    input_df['wickets_remaining'] = wickets_remaining
    input_df['runs_target'] = target
    input_df['rrr'] = rrr
    input_df['crr'] = crr

    if f'batting_team_{batting_team}' in model_columns:
        input_df[f'batting_team_{batting_team}'] = 1
    if f'bowling_team_{bowling_team}' in model_columns:
        input_df[f'bowling_team_{bowling_team}'] = 1
    if f'city_{city}' in model_columns:
        input_df[f'city_{city}'] = 1

    try:
        # Get raw probability from model
        prob = model.predict_proba(input_df)[0]
        win_p = prob[1] * 100
        
        # --- 3. DYNAMIC PRACTICALITY FILTERS (The Fix) ---
        
        # A. Wicket Penalty Logic
        # If wickets lost > 6, start applying a 'Pressure Factor'
        if wickets_lost >= 5:
            # Exponentially decrease win chance as wickets fall
            # 7 down: 30% penalty, 8 down: 60% penalty, 9 down: 90% penalty
            wicket_penalty_factor = {5: 0.8,6: 0.72,7: 0.63, 8: 0.45, 9: 0.19}
            win_p *= wicket_penalty_factor.get(wickets_lost, 1.0)
        
        # B. The "Impossible Chase" Filter
        # Even if the model is optimistic, RRR > 18 with 7+ wickets is nearly impossible
        if rrr > 18 and wickets_lost >= 5:
            win_p = min(win_p, 2.0)
            
        # C. Required Rate vs Wickets Remaining
        # If RRR is double the CRR and you have less than 3 wickets left
        if rrr > (crr * 2) and wickets_remaining <= 2:
            win_p *= 0.5

        # D. Smooth the final output
        final_win_p = round(max(0.0, min(100.0, win_p)), 1)
        return final_win_p

    except Exception as e:
        # Fallback to a basic RRR vs CRR logic if model fails
        return 50.0

if st.button("🔮 ANALYZE WIN CHANCE", use_container_width=True):
    win = get_prediction()
    loss = round(100 - win, 1)
    
    st.markdown("---")
    res1, res2 = st.columns(2)
    with res1:
        st.subheader(f"🔥 {batting_team}")
        st.title(f"{win}%")
        st.progress(int(win))
    with res2:
        st.subheader(f"🛡️ {bowling_team}")
        st.title(f"{loss}%")
        st.progress(int(loss))

st.markdown("---")
st.caption("Deepak's Professional Predictor | Multi-Feature Accuracy Mode")



#############################################################################################################################


