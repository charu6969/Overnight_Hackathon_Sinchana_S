import pandas as pd
import joblib
import os
from datetime import datetime
import numpy as np

# Import layer modules
from rules_engine import get_rule_score, get_social_engineering_score
from anomaly_model import get_ml_score
from graph_engine import update_graph, get_graph_score
from ai_fraud_module import get_screenshot_risk, get_voice_scam_risk
from persona_engine import get_user_risk_level

# --- Weights (As per requirement C) ---
ML_WEIGHT = 0.5
RULE_WEIGHT = 0.3
GRAPH_WEIGHT = 0.2
HIGH_RISK_THRESHOLD = 0.7

# --- New Weights for AI Innovation Scores (Adjust RULE_WEIGHT to accommodate) ---
# Total AI/Innovation Score Weight: 0.25 (to keep it significant)
AI_SCREENSHOT_WEIGHT = 0.08
AI_VOICE_WEIGHT = 0.08
AI_SOCIAL_WEIGHT = 0.09 
# Note: The rule_score now includes Device Intelligence and is weighted appropriately 

# Global in-memory transaction log (for ML feature calculation and velocity rules)
TRANSACTION_LOG = pd.DataFrame(columns=['transaction_id', 'timestamp', 'sender_id', 'receiver_id', 'amount'])

# Load the trained ML model once
ML_MODEL = None
try:
    ML_MODEL = joblib.load('./models/isolation_forest_model.pkl')
    # print("ML Model loaded successfully.")
except FileNotFoundError:
    print("WARNING: ML model not found. Run anomaly_model.py to train and save it.")
except Exception as e:
    print(f"Error loading ML model: {e}")


def run_fraud_detection(txn: dict) -> dict:
    """Runs the transaction through all detection layers and computes final score."""
    
    global TRANSACTION_LOG
    
    # 1. Update Graph
    update_graph(txn)
    
    # 2. Get User Persona (6)
    user_risk_level = get_user_risk_level(txn['sender_id'])
    txn['user_risk_level'] = user_risk_level # Add to txn for context
    
    # 3. Convert to DataFrame for ML/Rules
    current_txn_df = pd.DataFrame([txn])
    
    # --- Layer Scoring ---

    # 4. Rule-Based Score (Includes Device Intelligence (4) and Persona (6) factors)
    rule_score, rule_reasons = get_rule_score(txn, TRANSACTION_LOG)
    
    # 5. ML-Based Score (Anomaly Detection)
    ml_score = 0.0
    if ML_MODEL:
        ml_score = get_ml_score(current_txn_df, TRANSACTION_LOG, ML_MODEL)
    else:
        ml_score = 0.5 # Neutral score if model is missing

    # 6. Graph-Based Score (Fraud Ring Detection (8))
    graph_score, graph_reasons = get_graph_score(txn)
    
    # 7. AI Innovation Scores (1, 2, 3) - These add significant risk contributions
    
    # 7.1. Fake Screenshot Detection (1)
    screenshot_risk = get_screenshot_risk(txn)
    if screenshot_risk > 0.5:
        rule_reasons.append(f"AI: Fake Screenshot Risk ({screenshot_risk:.2f})")
    
    # 7.2. Voice Scam Detection (2)
    voice_scam_risk = get_voice_scam_risk(txn)
    if voice_scam_risk > 0.5:
        rule_reasons.append(f"AI: Voice Scam Pattern ({voice_scam_risk:.2f})")
        
    # 7.3. Social Engineering Cues (3)
    # We simulate a message content here for demonstration
    message_content = f"Hi {txn['sender_id']}, urgently click here for your refund of {txn['amount']}. PIN required." if random.random() < 0.1 else ""
    social_risk, social_reasons = get_social_engineering_score(message_content)
    
    if social_risk > 0.3:
        rule_reasons.extend(social_reasons)

    # 8. Final Scoring (Weighted Average + Innovation Contribution)
    
    # Base Score
    final_score = (ML_WEIGHT * ml_score) + (RULE_WEIGHT * rule_score) + (GRAPH_WEIGHT * graph_score)
    
    # Add Innovation Contributions
    final_score += (AI_SCREENSHOT_WEIGHT * screenshot_risk)
    final_score += (AI_VOICE_WEIGHT * voice_scam_risk)
    final_score += (AI_SOCIAL_WEIGHT * social_risk)
    
    # Ensure score remains 0-1
    final_score = np.clip(final_score, 0.0, 1.0)
    
    # 9. Flagging
    is_high_risk = final_score >= HIGH_RISK_THRESHOLD
    
    # 10. Update In-Memory Log
    log_txn = current_txn_df[['transaction_id', 'timestamp', 'sender_id', 'receiver_id', 'amount']].copy()
    TRANSACTION_LOG = pd.concat([TRANSACTION_LOG, log_txn], ignore_index=True)
    TRANSACTION_LOG = TRANSACTION_LOG.tail(10000) 
    
    # 11. Prepare Output
    all_reasons = rule_reasons + graph_reasons
    
    # Add ml explanation based on score magnitude (7)
    if ml_score > 0.7:
        all_reasons.append(f"ML Anomaly Detected (Velocity/Amount Outlier)")
        
    result = {
        'transaction_id': txn['transaction_id'],
        'timestamp': txn['timestamp'],
        'sender_id': txn['sender_id'],
        'receiver_id': txn['receiver_id'],
        'amount': txn['amount'],
        'rule_score': rule_score,
        'ml_score': ml_score,
        'graph_score': graph_score,
        'final_score': final_score,
        'is_high_risk': is_high_risk,
        'reasons': all_reasons,
        'user_risk_level': user_risk_level,
        'location': txn['location']
    }
    
    return result

if __name__ == '__main__':
    print("Fraud Score Service loaded.")