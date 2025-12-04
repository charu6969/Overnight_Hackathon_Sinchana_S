import streamlit as st
import pandas as pd
import time
import queue
import threading
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np

# --- 1. STREAMLIT PAGE CONFIG (MUST BE THE FIRST ST COMMAND) ---
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="FraudSight PRO")
st.title("🛡️ FraudSight PRO: UPI Transaction Risk Console (Hackathon)")
st.markdown("""
<style>
/* Custom Dark Theme Styling */
.stAlert > div { border-left: 5px solid #FF4B4B; }
.stProgress > div > div > div > div { background-color: #4CAF50; }
</style>
""", unsafe_allow_html=True) 
st.markdown("---")


# --- 2. PROJECT MODULE IMPORTS (Moved after page config) ---
# Importing these here ensures they load *after* the Streamlit context is initialized.
try:
    from simulator import run_simulator, USER_IDS
    from fraud_score_service import run_fraud_detection, GRAPH_WEIGHT, ML_WEIGHT, RULE_WEIGHT, HIGH_RISK_THRESHOLD
    from graph_engine import FRAUD_GRAPH
    from persona_engine import setup_user_personas, get_user_risk_level
    from ai_fraud_module import get_social_engineering_risk
except ImportError as e:
    st.error(f"FATAL SETUP ERROR: Cannot load project module. Did you save all .py files? Error: {e}")
    st.stop()


# --- 3. INITIAL SETUP CHECKS (Must run before UI elements) ---

# Run persona setup if data doesn't exist
if not os.path.exists('./data/user_personas.csv'):
    st.warning("User Persona data not found. Setting up 1000 user profiles now...")
    try:
        setup_user_personas(USER_IDS)
    except NameError:
        # Fallback if USER_IDS is not available due to simulator import error
        setup_user_personas([f'U{i:04d}' for i in range(1, 1001)])
    

# --- Session State Initialization ---
if 'transaction_queue' not in st.session_state:
    st.session_state.transaction_queue = queue.Queue()
if 'transaction_df' not in st.session_state:
    st.session_state.transaction_df = pd.DataFrame(
        columns=[
            'transaction_id', 'timestamp', 'sender_id', 'receiver_id', 'amount', 
            'rule_score', 'ml_score', 'graph_score', 'final_score', 'is_high_risk', 
            'reasons', 'user_risk_level', 'simulated_action', 'original_timestamp', 'location'
        ]
    )
if 'simulator_running' not in st.session_state:
    st.session_state.simulator_running = False

# Placeholder coordinates for Map Visualization (5)
CITY_COORDS_MAP = {
    'Mumbai': (19.0760, 72.8777),
    'Delhi': (28.7041, 77.1025),
    'Bangalore': (12.9716, 77.5946),
    'Chennai': (13.0827, 80.2707),
    'Pune': (18.5204, 73.8567)
}


# --- Core Logic Functions ---

def start_simulator_thread():
    """Starts the simulator in a separate thread."""
    if not st.session_state.simulator_running:
        st.session_state.simulator_running = True
        thread = threading.Thread(target=run_simulator, args=(st.session_state.transaction_queue,))
        thread.daemon = True
        thread.start()
        st.success("Transaction Simulator Started!")

def process_queue():
    """Pulls txns, runs fraud detection, and updates the log."""
    new_data = []
    
    for _ in range(10): 
        try:
            txn = st.session_state.transaction_queue.get_nowait()
            
            # Run the Fraud Detection Service
            result = run_fraud_detection(txn)
            
            # --- Innovation (9): Simulated Intervention/Action ---
            simulated_action = "Transaction Approved (Low Risk)"
            if result['is_high_risk']:
                reason_list = result['reasons']
                if any("New Device Usage" in r for r in reason_list) and any("High Txn Velocity" in r for r in reason_list):
                    simulated_action = "FREEZE ACCOUNT & NOTIFY USER (High Velocity/New Device)"
                elif any("Impossible Travel" in r for r in reason_list):
                    simulated_action = "REQUIRE STEP-UP AUTH (Selfie/OTP) (Impossible Travel)"
                elif result['ml_score'] > 0.8:
                     simulated_action = "TEMPORARY BLOCK (ML Anomaly)"
                else:
                    simulated_action = "FLAGGED FOR REVIEW"

            # Prepare data for dashboard table
            new_row = {
                'transaction_id': result['transaction_id'],
                'timestamp': result['timestamp'].strftime("%H:%M:%S"),
                'sender_id': result['sender_id'],
                'receiver_id': result['receiver_id'],
                'amount': f"{result['amount']:,.2f}",
                'rule_score': f"{result['rule_score']:.3f}",
                'ml_score': f"{result['ml_score']:.3f}",
                'graph_score': f"{result['graph_score']:.3f}",
                'final_score': f"{result['final_score']:.3f}",
                'is_high_risk': result['is_high_risk'],
                'reasons': ", ".join(result['reasons']),
                'user_risk_level': result['user_risk_level'],
                'simulated_action': simulated_action,
                'original_timestamp': result['timestamp'], 
                'location': txn['location']
            }
            new_data.append(new_row)
            
        except queue.Empty:
            break
        except Exception as e:
            print(f"Error processing transaction: {e}") # Debugging aid
            

    if new_data:
        new_df = pd.DataFrame(new_data)
        st.session_state.transaction_df = pd.concat([new_df, st.session_state.transaction_df], ignore_index=True).head(1000)


# --- Sidebar: Digital Literacy Companion (Game Changer) ---
with st.sidebar:
    st.header("🤖 Digital Literacy Companion")
    st.info("Ask me about UPI scams, PINS, or safety!")
    user_query = st.text_input("Your Query:")
    
    if user_query:
        query = user_query.lower()
        if "pin" in query or "otp" in query:
            st.markdown("🚨 **CRITICAL WARNING:** **NEVER** share your UPI PIN or OTP. You only use the PIN to **SEND** money. No one needs it for a refund or activation.")
        elif "refund" in query or "scam" in query:
            st.markdown("⚠️ **SCAM ALERT:** If someone sends you a link or QR code for a 'refund,' it is a **SCAM** designed to pull money from your account. Refunds happen automatically or via push notification.")
        elif "verification" in query or "activate" in query:
            st.markdown("🛡️ Banks or UPI systems will **NEVER** call you asking for verification details or account activation. Hang up immediately if they ask for sensitive data.")
        else:
            st.markdown("I am here to help. Remember the golden rule: **Never share your PIN!**")


# --- UI Layout ---

# Start/Stop Simulator Button
if not st.session_state.simulator_running:
    if st.button("▶️ Start Real-time Simulator"):
        start_simulator_thread()
else:
    st.success("Simulator is running, processing transactions every second.")

placeholder = st.empty()
process_queue()

col1, col2 = st.columns([3, 2])

# --- Column 1: Live Transactions Table & Alerts ---
with col1:
    st.header("Live Transaction Stream & Interventions (9)")
    
    # 1. Alerts Panel (Includes Innovation 9: Preventive Actions)
    st.subheader("🚨 HIGH-RISK ALERTS & INTERVENTIONS")
    high_risk_txns = st.session_state.transaction_df[st.session_state.transaction_df['is_high_risk'] == True].head(5)
    
    if high_risk_txns.empty:
        st.info("No high-risk transactions detected recently.")
    else:
        for _, row in high_risk_txns.iterrows():
            st.error(f"""
            **HIGH RISK T-ID: {row['transaction_id']}** - **Amount:** ₹{row['amount']} | **Score:** {row['final_score']}
            - **Intervention:** **{row['simulated_action']}** - **Persona/Location:** Risk Lvl {row['user_risk_level']} / {row['location']}
            - **Reasons (7):** {row['reasons']}
            """)

    st.subheader("🧾 Latest Transactions")
    
    # 2. Live Table (Color-coded)
    def color_score(val):
        """Color codes the final_score column."""
        try:
            score = float(val)
            if score >= HIGH_RISK_THRESHOLD:
                return 'background-color: #FF4B4B; color: white'
            elif score >= 0.5:
                return 'background-color: #FFA500; color: black'
            else:
                return 'background-color: #00FF7F; color: black'
        except:
            return ''

    # Show the table, dropping technical internal columns
    display_df = st.session_state.transaction_df.drop(columns=['original_timestamp', 'is_high_risk', 'simulated_action', 'location', 'user_risk_level'], errors='ignore')
    styled_df = display_df.head(15).style.applymap(color_score, subset=['final_score'])
    st.dataframe(styled_df, hide_index=True, use_container_width=True)

# --- Column 2: Charts and Explainability ---
with col2:
    st.header("Analytics & Investigation")
    
    if not st.session_state.transaction_df.empty:
        plot_df = st.session_state.transaction_df.copy()
        # Convert score columns to numeric (handling the f-string formatting done earlier)
        plot_df['final_score_float'] = pd.to_numeric(plot_df['final_score'], errors='coerce')
        plot_df['rule_score_float'] = pd.to_numeric(plot_df['rule_score'], errors='coerce')
        plot_df['ml_score_float'] = pd.to_numeric(plot_df['ml_score'], errors='coerce')
        plot_df['graph_score_float'] = pd.to_numeric(plot_df['graph_score'], errors='coerce')
        
        # --- Innovation (7): ML Explainability Panel (Contribution Chart) ---
        st.subheader("⚖️ ML Explainability (Score Contribution)")
        contributions = {
            'Layer': ['ML Anomaly', 'Rule Heuristics', 'Graph Analysis'],
            'Weighted Contribution': [
                plot_df['ml_score_float'].mean() * ML_WEIGHT, 
                plot_df['rule_score_float'].mean() * RULE_WEIGHT, 
                plot_df['graph_score_float'].mean() * GRAPH_WEIGHT
            ]
        }
        contrib_df = pd.DataFrame(contributions)
        
        fig_contrib = px.bar(contrib_df, x='Layer', y='Weighted Contribution',
                             height=200,
                             color='Layer',
                             color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'])
        fig_contrib.update_layout(margin={"t":30, "b":10, "l":10, "r":10})
        st.plotly_chart(fig_contrib, use_container_width=True)

        # --- Innovation (10): Post-Fraud Recovery/Investigation Module ---
        st.subheader("🕵️ Investigation Module (Case Management)")
        flagged_txn = high_risk_txns.head(1)
        if not flagged_txn.empty:
            st.warning(f"**Case ID:** FRD-{flagged_txn['transaction_id'].iloc[0].split('_')[0]}")
            st.json({
                "SuspectSender": flagged_txn['sender_id'].iloc[0],
                "RecommendedAction": flagged_txn['simulated_action'].iloc[0] + " | Initiate Fraud Timeline & Block Account",
                "FlaggedReasons": flagged_txn['reasons'].iloc[0]
            })
        else:
            st.info("Module ready. A fraud report will generate upon the next HIGH-RISK alert.")

st.markdown("---")

# --- Global Visualization Section ---

# --- Innovation (5): Fraud HeatMap (Real-Time India Map Visualization) ---
st.header("🌍 Real-Time Fraud HeatMap")
if not st.session_state.transaction_df.empty:
    map_df = st.session_state.transaction_df.copy()
    
    map_df['lat'] = map_df['location'].map(lambda x: CITY_COORDS_MAP.get(x, (20.5937, 78.9629))[0])
    map_df['lon'] = map_df['location'].map(lambda x: CITY_COORDS_MAP.get(x, (20.5937, 78.9629))[1])
    
    high_risk_map_df = map_df[map_df['is_high_risk'] == True].tail(200)

    if not high_risk_map_df.empty:
        high_risk_map_df['final_score_float'] = pd.to_numeric(high_risk_map_df['final_score'], errors='coerce') 
        
        fig_map = px.scatter_mapbox(high_risk_map_df, 
                                    lat="lat", 
                                    lon="lon",
                                    color="final_score_float",
                                    size="final_score_float", 
                                    color_continuous_scale=px.colors.sequential.Reds,
                                    zoom=4,
                                    center={"lat": 23.0, "lon": 78.0}, 
                                    mapbox_style="carto-darkmatter", 
                                    title="Fraud Hotspots (Color=Risk Score)"
                                    )
        fig_map.update_layout(height=400, margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Map is waiting for high-risk transactions to plot fraud hotspots.")


# --- Innovation (8): Fraud Ring Detection (Graph + Visualization) ---
st.subheader("🕸️ Fraud Ring Network Analysis")

if FRAUD_GRAPH.number_of_nodes() > 50:
    try:
        recent_flagged_senders = st.session_state.transaction_df[
            st.session_state.transaction_df['is_high_risk'] == True
        ]['sender_id'].unique()[:10]

        subgraph_nodes = set(recent_flagged_senders)
        for node in subgraph_nodes.copy():
            if node in FRAUD_GRAPH:
                subgraph_nodes.update(list(FRAUD_GRAPH.neighbors(node)))
                subgraph_nodes.update(list(FRAUD_GRAPH.predecessors(node)))
        
        nodes_to_draw = list(subgraph_nodes)[:30]
        subgraph = FRAUD_GRAPH.subgraph(nodes_to_draw)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        pos = nx.spring_layout(subgraph, k=0.3, iterations=10)
        
        node_colors = ['red' if node in recent_flagged_senders else 'lightblue' for node in subgraph.nodes()]

        nx.draw_networkx_nodes(subgraph, pos, node_size=200, node_color=node_colors, ax=ax)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(subgraph, pos, font_size=8, ax=ax)
        
        ax.set_title(f"Fraud Ring Subgraph ({len(subgraph)} Nodes) - Highlighted High-Risk")
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error drawing graph: {e}")

else:
    st.info(f"Graph building in progress. {FRAUD_GRAPH.number_of_nodes()} nodes processed. Visualization requires more data (>50 nodes).")
    
# --- Auto-Refresh Rerun ---
# --- Auto-Refresh Rerun ---
if st.session_state.simulator_running:
    placeholder.empty()
    time.sleep(2)
    # REPLACE THIS: st.experimental_rerun()
    st.rerun() # <-- CORRECTED FUNCTION