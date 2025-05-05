import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="RFQ Dealer Simulator", layout="wide")

# -------------------- Load Model --------------------
model = joblib.load("lgb_model.pkl")  # Pre-trained LightGBM model

# -------------------- Load Data --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("rfqs.csv", parse_dates=['DateTime'])
    df['signed_spread'] = (df['price'] - df['mid']) * (-df['side'])
    return df[['DateTime', 'instrument', 'price', 'mid', 'vol(MM)', 'dv01', 'num_dealers', 'side', 'signed_spread', 'won', 'client']]

rfq_data = load_data()

# -------------------- Initialize State --------------------
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        'RFQ Time', 'Instrument', 'Side', 'Volume', 'Quote Spread', 'Win Prob', 'Outcome', 'Revenue'
    ])
if 'current_rfq_idx' not in st.session_state:
    st.session_state.current_rfq_idx = np.random.randint(0, len(rfq_data))

current_rfq = rfq_data.iloc[st.session_state.current_rfq_idx]

# -------------------- Precompute Prediction Grid --------------------
quote_spread = 0.0
spread_grid = np.linspace(-1.5, 1.5, 100)
signed_spread_grid = spread_grid * -current_rfq['side']

full_feature_list = model.feature_name_
client_cols = [col for col in full_feature_list if col.startswith('client_')]
base_cols = [col for col in full_feature_list if col not in client_cols]

sim_df = pd.DataFrame({
    'signed_spread': signed_spread_grid,
    'spread_per_dv01': signed_spread_grid / current_rfq['dv01'] if current_rfq['dv01'] != 0 else 0,
    'log_vol': np.log(current_rfq['vol(MM)']) if current_rfq['vol(MM)'] > 0 else 0,
    'hour': current_rfq['DateTime'].hour,
    'dayofweek': current_rfq['DateTime'].weekday(),
    'num_dealers': current_rfq['num_dealers']
})

client_id = current_rfq.get('client', None)
client_onehot = pd.DataFrame(
    {col: int(col == f'client_{client_id}') for col in client_cols},
    index=sim_df.index
)
sim_df = pd.concat([sim_df, client_onehot], axis=1)
sim_df = sim_df.reindex(columns=full_feature_list, fill_value=0)

win_probs = model.predict_proba(sim_df)[:, 1]
expected_revenue = win_probs * spread_grid * current_rfq['vol(MM)']
opt_idx = np.argmax(expected_revenue)
opt_spread = spread_grid[opt_idx]
opt_win_prob = win_probs[opt_idx]
opt_revenue = expected_revenue[opt_idx]

# -------------------- Layout --------------------
st.title("📈 Dealer RFQ Simulation (Historical)")
st.markdown("Respond to real RFQs using your pricing model. Quote a spread, see win probability, and track your results.")

st.divider()

# -------------------- Two-Column Layout --------------------
left, right = st.columns([3, 4])

with left:
    st.subheader("📨 RFQ Details")
    st.markdown(f"""
    - **Time**: {current_rfq['DateTime']}
    - **Instrument**: `{current_rfq['instrument']}`
    - **Side**: `{"Buy" if current_rfq['side'] == -1 else "Sell"}`
    - **Volume**: `{current_rfq['vol(MM)']} MM`
    - **DV01**: `{current_rfq['dv01']}`
    - **Mid Price**: `{current_rfq['mid']:.3f}`
    - **# Dealers**: `{current_rfq['num_dealers']}`
    """)

    st.markdown("### 💰 Expected Revenue vs Spread")
    fig_revenue = go.Figure()
    fig_revenue.add_trace(go.Scatter(x=spread_grid, y=expected_revenue, mode='lines', name='Expected Revenue', line=dict(color='green')))
    fig_revenue.add_vline(x=quote_spread, line=dict(color='orange', dash='dash'), annotation_text="Your Quote")
    fig_revenue.add_vline(x=opt_spread, line=dict(color='blue', dash='dot'), annotation_text=f"Optimal: {opt_spread:.3f}")
    fig_revenue.update_layout(height=400, xaxis_title="Quoted Spread", yaxis_title="Expected Revenue")
    st.plotly_chart(fig_revenue, use_container_width=True)

with right:
    st.subheader("💬 Submit Your Quote")
    quote_spread = st.slider("Quoted Spread", -1.5, 1.5, 0.0, step=0.01)
    signed_spread = quote_spread * -current_rfq['side']

    st.markdown("### 📈 Win Probability vs Spread")
    fig_win = go.Figure()
    fig_win.add_trace(go.Scatter(x=spread_grid, y=win_probs, mode='lines', name='Win Probability', line=dict(color='blue')))
    fig_win.add_vline(x=quote_spread, line=dict(color='orange', dash='dash'), annotation_text="Your Quote")
    fig_win.add_vline(x=opt_spread, line=dict(color='green', dash='dot'), annotation_text=f"Optimal: {opt_spread:.3f}")
    fig_win.update_layout(height=400, xaxis_title="Quoted Spread", yaxis_title="Win Probability")
    st.plotly_chart(fig_win, use_container_width=True)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

    with col_btn1:
        if st.button("📤 Submit Quote"):
            spread_per_dv01 = signed_spread / current_rfq['dv01'] if current_rfq['dv01'] != 0 else 0
            log_vol = math.log(current_rfq['vol(MM)']) if current_rfq['vol(MM)'] > 0 else 0
            model_input = pd.DataFrame([{
                'signed_spread': signed_spread,
                'spread_per_dv01': spread_per_dv01,
                'log_vol': log_vol,
                'hour': current_rfq['DateTime'].hour,
                'dayofweek': current_rfq['DateTime'].weekday(),
                'num_dealers': current_rfq['num_dealers']
            }])
            client_onehot = pd.DataFrame(
                {col: int(col == f'client_{client_id}') for col in client_cols},
                index=[0]
            )
            model_input = pd.concat([model_input, client_onehot], axis=1)
            model_input = model_input.reindex(columns=full_feature_list, fill_value=0)
            win_prob = model.predict_proba(model_input)[0, 1]
            expected_revenue = win_prob * quote_spread * current_rfq['vol(MM)']
            won = np.random.rand() < win_prob
            revenue = quote_spread * current_rfq['vol(MM)'] if won else 0
            new_entry = {
                'RFQ Time': current_rfq['DateTime'],
                'Instrument': current_rfq['instrument'],
                'Side': "Buy" if current_rfq['side'] == -1 else "Sell",
                'Volume': current_rfq['vol(MM)'],
                'Quote Spread': quote_spread,
                'Win Prob': round(win_prob, 3),
                'Outcome': "✅ Win" if won else "❌ Loss",
                'Revenue': round(revenue, 3)
            }
            st.session_state.history = pd.concat([
                st.session_state.history,
                pd.DataFrame([new_entry])
            ], ignore_index=True)
            st.success(f"{'✅ You WON' if won else '❌ You LOST'} this RFQ")
            st.markdown(f"""
        - **Win Probability**: `{win_prob:.2%}`
        - **Expected Revenue**: `{expected_revenue:.3f}`
        - **Actual Revenue**: `{revenue:.3f}`
        - **Optimal Spread Used**: `{opt_spread:.3f}`
        """)

    with col_btn2:
        if st.button("🎯 Optimal Quote"):
            signed_spread = opt_spread * -current_rfq['side']
            spread_per_dv01 = signed_spread / current_rfq['dv01'] if current_rfq['dv01'] != 0 else 0
            log_vol = math.log(current_rfq['vol(MM)']) if current_rfq['vol(MM)'] > 0 else 0
            model_input = pd.DataFrame([{
                'signed_spread': signed_spread,
                'spread_per_dv01': spread_per_dv01,
                'log_vol': log_vol,
                'hour': current_rfq['DateTime'].hour,
                'dayofweek': current_rfq['DateTime'].weekday(),
                'num_dealers': current_rfq['num_dealers']
            }])
            client_onehot = pd.DataFrame(
                {col: int(col == f'client_{client_id}') for col in client_cols},
                index=[0]
            )
            model_input = pd.concat([model_input, client_onehot], axis=1)
            model_input = model_input.reindex(columns=full_feature_list, fill_value=0).astype(float)
            win_prob = model.predict_proba(model_input)[0, 1]
            expected_revenue = win_prob * opt_spread * current_rfq['vol(MM)']
            won = np.random.rand() < win_prob
            revenue = opt_spread * current_rfq['vol(MM)'] if won else 0
            new_entry = {
                'RFQ Time': current_rfq['DateTime'],
                'Instrument': current_rfq['instrument'],
                'Side': "Buy" if current_rfq['side'] == -1 else "Sell",
                'Volume': current_rfq['vol(MM)'],
                'Quote Spread': opt_spread,
                'Win Prob': round(win_prob, 3),
                'Outcome': "✅ Win" if won else "❌ Loss",
                'Revenue': round(revenue, 3)
            }
            st.session_state.history = pd.concat([
                st.session_state.history,
                pd.DataFrame([new_entry])
            ], ignore_index=True)
            st.success(f"🎯 You submitted the OPTIMAL quote — {'✅ Win' if won else '❌ Loss'}")
            st.markdown(f"""
        - **Win Probability**: `{win_prob:.2%}`
        - **Expected Revenue**: `{expected_revenue:.3f}`
        - **Actual Revenue**: `{revenue:.3f}`
        - **Optimal Spread Used**: `{opt_spread:.3f}`
        """)

    with col_btn3:
        if st.button("➡️ Next Quote"):
            st.session_state.current_rfq_idx = np.random.randint(0, len(rfq_data))
            st.rerun()

# -------------------- Quote History --------------------
st.divider()
st.subheader("📄 Quote History")

if not st.session_state.history.empty:
    st.dataframe(st.session_state.history, use_container_width=True)
    total_quotes = len(st.session_state.history)
    total_wins = (st.session_state.history['Outcome'] == "✅ Win").sum()
    total_revenue = st.session_state.history['Revenue'].sum()
    colA, colB, colC = st.columns(3)
    colA.metric("Total RFQs", total_quotes)
    colB.metric("Win Rate", f"{total_wins / total_quotes:.2%}")
    colC.metric("Total Revenue", f"{total_revenue:.3f}")
else:
    st.info("No quotes submitted yet.")














        