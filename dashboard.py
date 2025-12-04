import streamlit as st
import pandas as pd
import time
import queue
import threading
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np

# --- 1. STREAMLIT PAGE CONFIG (MUST BE THE FIRST ST COMMAND) ---
st.set_page_config(
    layout="wide", 
    initial_sidebar_state="expanded", 
    page_title="FraudSight PRO",
    page_icon="🛡️"
)

# --- Enhanced Custom Styling ---
st.markdown("""
<style>
/* Dark Theme with Gradient Background */
.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}

/* Custom Alert Styling */
.stAlert > div { 
    border-left: 5px solid #FF4B4B;
    background-color: rgba(255, 75, 75, 0.1);
    backdrop-filter: blur(10px);
}

/* Progress Bar Styling */
.stProgress > div > div > div > div { 
    background: linear-gradient(90deg, #4CAF50, #45a049);
}

/* Card Styling */
.metric-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
}

/* Sidebar Styling */
.css-1d391kg {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
}

/* Header Styling */
h1, h2, h3 {
    color: #ffffff;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

/* Button Styling */
.stButton>button {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: bold;
    box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4);
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.6);
}

/* Dataframe Styling */
.dataframe {
    border-radius: 10px;
    overflow: hidden;
}

/* Animation for new alerts */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

.alert-pulse {
    animation: pulse 2s infinite;
}

/* Success/Info Box Styling */
.stSuccess, .stInfo {
    background: rgba(76, 175, 80, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 10px;
    border-left: 4px solid #4CAF50;
}
</style>
""", unsafe_allow_html=True)

# --- Header Section with Enhanced Design ---
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.markdown("# 🛡️")
with col_title:
    st.markdown("""
    # FraudSight PRO
    ### AI-Powered UPI Transaction Risk Intelligence System
    """)
st.markdown("---")

# --- 2. PROJECT MODULE IMPORTS ---
try:
    from simulator import run_simulator, USER_IDS
    from fraud_score_service import run_fraud_detection, GRAPH_WEIGHT, ML_WEIGHT, RULE_WEIGHT, HIGH_RISK_THRESHOLD
    from graph_engine import FRAUD_GRAPH
    from persona_engine import setup_user_personas, get_user_risk_level
    from ai_fraud_module import get_social_engineering_risk
    st.success("✅ All modules loaded successfully!")
except ImportError as e:
    st.error(f"❌ FATAL SETUP ERROR: Cannot load project module. Error: {e}")
    st.info("💡 Please ensure all Python files are in the correct directory and dependencies are installed.")
    st.stop()

# --- 3. INITIAL SETUP CHECKS ---
if not os.path.exists('./data/user_personas.csv'):
    with st.spinner("⚙️ Setting up user personas (1000 profiles)..."):
        try:
            setup_user_personas(USER_IDS)
            st.success("✅ User personas created successfully!")
        except NameError:
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

if 'total_fraud_prevented' not in st.session_state:
    st.session_state.total_fraud_prevented = 0

if 'total_amount_saved' not in st.session_state:
    st.session_state.total_amount_saved = 0.0

# City coordinates for map visualization
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
        st.success("🚀 Transaction Simulator Started!")
        st.balloons()

def process_queue():
    """Pulls transactions, runs fraud detection, and updates the log."""
    new_data = []
    
    for _ in range(10): 
        try:
            txn = st.session_state.transaction_queue.get_nowait()
            
            # Run the Fraud Detection Service
            result = run_fraud_detection(txn)
            
            # --- Simulated Intervention/Action ---
            simulated_action = "✅ Transaction Approved (Low Risk)"
            if result['is_high_risk']:
                reason_list = result['reasons']
                if any("New Device Usage" in r for r in reason_list) and any("High Txn Velocity" in r for r in reason_list):
                    simulated_action = "🔒 FREEZE ACCOUNT & NOTIFY USER (High Velocity/New Device)"
                elif any("Impossible Travel" in r for r in reason_list):
                    simulated_action = "🔐 REQUIRE STEP-UP AUTH (Selfie/OTP) (Impossible Travel)"
                elif result['ml_score'] > 0.8:
                    simulated_action = "⛔ TEMPORARY BLOCK (ML Anomaly)"
                else:
                    simulated_action = "⚠️ FLAGGED FOR REVIEW"
                
                # Update fraud prevention stats
                st.session_state.total_fraud_prevented += 1
                st.session_state.total_amount_saved += result['amount']

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
            st.error(f"⚠️ Error processing transaction: {e}")

    if new_data:
        new_df = pd.DataFrame(new_data)
        st.session_state.transaction_df = pd.concat([new_df, st.session_state.transaction_df], ignore_index=True).head(1000)

# --- Enhanced Sidebar: Digital Literacy Companion ---
with st.sidebar:
    st.markdown("## 🤖 Digital Literacy Companion")
    st.markdown("---")
    
    # Quick Stats
    st.metric("🛡️ Frauds Prevented", st.session_state.total_fraud_prevented)
    st.metric("💰 Amount Saved", f"₹{st.session_state.total_amount_saved:,.2f}")
    
    st.markdown("---")
    st.info("💡 Ask me about UPI scams, PINs, or safety!")
    
    user_query = st.text_input("Your Safety Query:", placeholder="e.g., What is a PIN?")
    
    if user_query:
        query = user_query.lower()
        if "pin" in query or "otp" in query:
            st.error("""
            🚨 **CRITICAL WARNING:** 
            
            **NEVER** share your UPI PIN or OTP with anyone!
            
            ✅ You only use PIN to **SEND** money
            ❌ No one needs it for refunds or activation
            """)
        elif "refund" in query or "scam" in query:
            st.warning("""
            ⚠️ **SCAM ALERT:**
            
            If someone sends you a link or QR code for a 'refund,' it is a **SCAM** designed to pull money from your account.
            
            ✅ Real refunds happen automatically
            ❌ Never click suspicious refund links
            """)
        elif "verification" in query or "activate" in query:
            st.warning("""
            🛡️ **VERIFICATION SCAM:**
            
            Banks or UPI systems will **NEVER** call you asking for:
            - Account verification
            - PIN/OTP
            - Account activation
            
            ☎️ Hang up immediately if they ask for sensitive data!
            """)
        else:
            st.info("""
            I'm here to help keep you safe! 
            
            🔐 **Golden Rule:** Never share your PIN or OTP
            
            Ask me anything about:
            - UPI safety
            - Scam detection
            - Secure transactions
            """)
    
    st.markdown("---")
    st.markdown("### 📚 Quick Safety Tips")
    st.markdown("""
    - 🔒 Never share PIN/OTP
    - ✅ Verify receiver details
    - 🚫 Don't click unknown links
    - 📱 Use official apps only
    - 🔐 Enable 2FA authentication
    """)

# --- Main Dashboard Area ---

# Control Panel
col_control1, col_control2, col_control3 = st.columns([2, 1, 1])

with col_control1:
    if not st.session_state.simulator_running:
        if st.button("▶️ Start Real-time Fraud Detection Simulator", use_container_width=True):
            start_simulator_thread()
    else:
        st.success("🟢 Simulator Running | Processing transactions in real-time...")

with col_control2:
    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.rerun()

with col_control3:
    total_txns = len(st.session_state.transaction_df)
    st.metric("📊 Total Txns", total_txns)

placeholder = st.empty()
process_queue()

# --- Key Metrics Dashboard ---
if not st.session_state.transaction_df.empty:
    st.markdown("### 📈 Real-Time Metrics")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    total_txns = len(st.session_state.transaction_df)
    high_risk_count = len(st.session_state.transaction_df[st.session_state.transaction_df['is_high_risk'] == True])
    
    with metric_col1:
        st.metric(
            "🔍 Total Transactions", 
            total_txns,
            delta=f"+{10 if st.session_state.simulator_running else 0}",
            delta_color="normal"
        )
    
    with metric_col2:
        st.metric(
            "🚨 High Risk Detected", 
            high_risk_count,
            delta=f"{(high_risk_count/total_txns*100):.1f}% of total" if total_txns > 0 else "0%",
            delta_color="inverse"
        )
    
    with metric_col3:
        st.metric(
            "🛡️ Frauds Prevented",
            st.session_state.total_fraud_prevented,
            delta=f"₹{st.session_state.total_amount_saved:,.0f} saved"
        )
    
    with metric_col4:
        avg_risk = st.session_state.transaction_df['final_score'].astype(float).mean()
        st.metric(
            "📊 Avg Risk Score",
            f"{avg_risk:.2%}",
            delta="Lower is better",
            delta_color="inverse"
        )

st.markdown("---")

# --- Main Content Layout ---
col1, col2 = st.columns([3, 2])

# --- Column 1: Live Transactions Table & Alerts ---
with col1:
    st.markdown("## 🔴 Live Transaction Stream & Interventions")
    
    # High-Risk Alerts Panel
    st.markdown("### 🚨 HIGH-RISK ALERTS & INTERVENTIONS")
    high_risk_txns = st.session_state.transaction_df[st.session_state.transaction_df['is_high_risk'] == True].head(5)
    
    if high_risk_txns.empty:
        st.info("✅ No high-risk transactions detected recently. System is monitoring...")
    else:
        for idx, row in high_risk_txns.iterrows():
            with st.expander(f"🚨 HIGH RISK: {row['transaction_id']} | ₹{row['amount']} | Score: {row['final_score']}", expanded=True):
                alert_col1, alert_col2 = st.columns([2, 1])
                
                with alert_col1:
                    st.markdown(f"""
                    **Transaction Details:**
                    - **Sender:** `{row['sender_id']}`
                    - **Receiver:** `{row['receiver_id']}`
                    - **Location:** 📍 {row['location']}
                    - **Time:** ⏰ {row['timestamp']}
                    - **User Risk Level:** {'🔴 High' if row['user_risk_level'] == 1 else '🟡 Medium' if row['user_risk_level'] == 2 else '🟢 Low'}
                    """)
                
                with alert_col2:
                    st.markdown(f"""
                    **Risk Scores:**
                    - ML: `{row['ml_score']}`
                    - Rule: `{row['rule_score']}`
                    - Graph: `{row['graph_score']}`
                    """)
                
                st.error(f"**🔒 INTERVENTION:** {row['simulated_action']}")
                st.warning(f"**⚠️ Detection Reasons:** {row['reasons']}")
                
                st.markdown("---")

    # Latest Transactions Table
    st.markdown("### 🧾 Latest Transactions (Live Feed)")
    
    def color_score(val):
        """Color codes the final_score column."""
        try:
            score = float(val)
            if score >= HIGH_RISK_THRESHOLD:
                return 'background-color: #FF4B4B; color: white; font-weight: bold;'
            elif score >= 0.5:
                return 'background-color: #FFA500; color: black; font-weight: bold;'
            else:
                return 'background-color: #4CAF50; color: white; font-weight: bold;'
        except:
            return ''

    # Show the table
    if not st.session_state.transaction_df.empty:
        display_df = st.session_state.transaction_df.drop(
            columns=['original_timestamp', 'is_high_risk', 'simulated_action', 'location', 'user_risk_level'], 
            errors='ignore'
        )
        styled_df = display_df.head(20).style.applymap(color_score, subset=['final_score'])
        st.dataframe(styled_df, hide_index=True, use_container_width=True, height=400)
    else:
        st.info("⏳ Waiting for transactions... Start the simulator to see live data.")

# --- Column 2: Analytics & Investigation ---
with col2:
    st.markdown("## 📊 Analytics & Investigation")
    
    if not st.session_state.transaction_df.empty:
        plot_df = st.session_state.transaction_df.copy()
        plot_df['final_score_float'] = pd.to_numeric(plot_df['final_score'], errors='coerce')
        plot_df['rule_score_float'] = pd.to_numeric(plot_df['rule_score'], errors='coerce')
        plot_df['ml_score_float'] = pd.to_numeric(plot_df['ml_score'], errors='coerce')
        plot_df['graph_score_float'] = pd.to_numeric(plot_df['graph_score'], errors='coerce')
        
        # ML Explainability Panel
        st.markdown("### ⚖️ ML Explainability (Score Contribution)")
        contributions = {
            'Layer': ['ML Anomaly', 'Rule Heuristics', 'Graph Analysis'],
            'Weighted Contribution': [
                plot_df['ml_score_float'].mean() * ML_WEIGHT, 
                plot_df['rule_score_float'].mean() * RULE_WEIGHT, 
                plot_df['graph_score_float'].mean() * GRAPH_WEIGHT
            ]
        }
        contrib_df = pd.DataFrame(contributions)
        
        fig_contrib = px.bar(
            contrib_df, 
            x='Layer', 
            y='Weighted Contribution',
            height=250,
            color='Layer',
            color_discrete_sequence=['#667eea', '#ff7f0e', '#2ca02c'],
            title="Detection Layer Contributions"
        )
        fig_contrib.update_layout(
            margin={"t":40, "b":10, "l":10, "r":10},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_contrib, use_container_width=True)

        # Risk Score Distribution
        st.markdown("### 📊 Risk Score Distribution")
        fig_dist = px.histogram(
            plot_df, 
            x='final_score_float',
            nbins=20,
            color_discrete_sequence=['#667eea'],
            title="Transaction Risk Distribution"
        )
        fig_dist.update_layout(
            margin={"t":40, "b":10, "l":10, "r":10},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis_title="Risk Score",
            yaxis_title="Count"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Investigation Module
        st.markdown("### 🕵️ Investigation Module (Case Management)")
        flagged_txn = high_risk_txns.head(1)
        if not flagged_txn.empty:
            case_id = f"FRD-{flagged_txn['transaction_id'].iloc[0].split('_')[0]}"
            st.warning(f"**📋 Active Case ID:** `{case_id}`")
            
            case_data = {
                "🎯 Suspect Sender": flagged_txn['sender_id'].iloc[0],
                "📍 Location": flagged_txn['location'].iloc[0],
                "💰 Amount": f"₹{flagged_txn['amount'].iloc[0]}",
                "⚠️ Risk Score": flagged_txn['final_score'].iloc[0],
                "🔒 Recommended Action": flagged_txn['simulated_action'].iloc[0],
                "🔍 Flagged Reasons": flagged_txn['reasons'].iloc[0]
            }
            
            st.json(case_data)
            
            if st.button("📥 Export Case Report", use_container_width=True):
                st.success("✅ Case report generated and ready for download!")
        else:
            st.info("✅ No active fraud cases. System ready for investigation.")

st.markdown("---")

# --- Global Visualization Section ---

# Fraud HeatMap
st.markdown("## 🌍 Real-Time Fraud HeatMap (Geographic Intelligence)")

if not st.session_state.transaction_df.empty:
    map_df = st.session_state.transaction_df.copy()
    
    map_df['lat'] = map_df['location'].map(lambda x: CITY_COORDS_MAP.get(x, (20.5937, 78.9629))[0])
    map_df['lon'] = map_df['location'].map(lambda x: CITY_COORDS_MAP.get(x, (20.5937, 78.9629))[1])
    
    high_risk_map_df = map_df[map_df['is_high_risk'] == True].tail(200)

    if not high_risk_map_df.empty:
        high_risk_map_df['final_score_float'] = pd.to_numeric(high_risk_map_df['final_score'], errors='coerce')
        
        fig_map = px.scatter_mapbox(
            high_risk_map_df, 
            lat="lat", 
            lon="lon",
            color="final_score_float",
            size="final_score_float",
            size_max=20,
            color_continuous_scale="Reds",
            zoom=4,
            center={"lat": 23.0, "lon": 78.0}, 
            mapbox_style="carto-darkmatter",
            hover_data={
                'sender_id': True,
                'amount': True,
                'location': True,
                'final_score_float': ':.2f'
            },
            title="🔥 Fraud Hotspots Across India (Real-Time)"
        )
        fig_map.update_layout(
            height=500, 
            margin={"r":0,"t":50,"l":0,"b":0},
            font=dict(color='white')
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("🗺️ Map is waiting for high-risk transactions to plot fraud hotspots...")
else:
    st.info("⏳ Start the simulator to see fraud geographic distribution...")

st.markdown("---")

# Fraud Ring Network Analysis
st.markdown("## 🕸️ Fraud Ring Network Analysis (Graph Intelligence)")

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
        
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        
        pos = nx.spring_layout(subgraph, k=0.5, iterations=20)
        
        node_colors = ['#FF4B4B' if node in recent_flagged_senders else '#4FC3F7' for node in subgraph.nodes()]

        nx.draw_networkx_nodes(subgraph, pos, node_size=400, node_color=node_colors, ax=ax, alpha=0.8)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.3, edge_color='white', arrows=True, arrowsize=15, ax=ax)
        nx.draw_networkx_labels(subgraph, pos, font_size=9, font_color='white', font_weight='bold', ax=ax)
        
        ax.set_title(f"Fraud Ring Subgraph ({len(subgraph)} Nodes) - Red: High-Risk Users", 
                    color='white', fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        st.pyplot(fig, use_container_width=True)
        
        st.success(f"✅ Detected {len(recent_flagged_senders)} potential fraud ring members with {subgraph.number_of_edges()} suspicious connections!")
        
    except Exception as e:
        st.error(f"⚠️ Error drawing graph: {e}")
else:
    progress = (FRAUD_GRAPH.number_of_nodes() / 50) * 100
    st.progress(progress / 100)
    st.info(f"🔄 Graph building in progress: {FRAUD_GRAPH.number_of_nodes()} nodes processed. Need 50+ nodes for visualization (currently {progress:.0f}% complete).")

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; padding: 20px; color: #888;'>
    <p><strong>FraudSight PRO v2.0</strong> | Powered by AI & Machine Learning</p>
    <p>🛡️ Protecting UPI transactions in real-time | Built for hackathon excellence</p>
</div>
""", unsafe_allow_html=True)

# Auto-Refresh Logic
if st.session_state.simulator_running:
    placeholder.empty()
    time.sleep(2)
    st.rerun()