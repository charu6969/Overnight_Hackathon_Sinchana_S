import pandas as pd
import joblib
import os
from datetime import datetime
import numpy as np
import random

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

# ⚡ LOWERED THRESHOLD: From 0.7 to 0.6 (more sensitive)
HIGH_RISK_THRESHOLD = 0.6  # Was 0.7

# --- New Weights for AI Innovation Scores ---
AI_SCREENSHOT_WEIGHT = 0.08
AI_VOICE_WEIGHT = 0.08
AI_SOCIAL_WEIGHT = 0.09

# Global in-memory transaction log
TRANSACTION_LOG = pd.DataFrame(columns=['transaction_id', 'timestamp', 'sender_id', 'receiver_id', 'amount'])

# Load the trained ML model once
ML_MODEL = None
try:
    ML_MODEL = joblib.load('./models/isolation_forest_model.pkl')
    print("✅ ML Model loaded successfully.")
except FileNotFoundError:
    print("⚠️ WARNING: ML model not found. Run: python anomaly_model.py")
except Exception as e:
    print(f"❌ Error loading ML model: {e}")


def run_fraud_detection(txn: dict) -> dict:
    """Runs the transaction through all detection layers and computes final score."""
    
    global TRANSACTION_LOG
    
    try:
        # 1. Update Graph
        update_graph(txn)
        
        # 2. Get User Persona
        user_risk_level = get_user_risk_level(txn['sender_id'])
        txn['user_risk_level'] = user_risk_level
        
        # 3. Convert to DataFrame for ML/Rules
        current_txn_df = pd.DataFrame([txn])
        
        # --- Layer Scoring ---

        # 4. Rule-Based Score
        try:
            rule_score, rule_reasons = get_rule_score(txn, TRANSACTION_LOG)
        except Exception as e:
            print(f"⚠️ Rule scoring error: {e}")
            rule_score, rule_reasons = 0.5, ["Rule scoring error"]
        
        # 5. ML-Based Score (Anomaly Detection)
        ml_score = 0.0
        if ML_MODEL and len(TRANSACTION_LOG) >= 10:  # Need some history
            try:
                ml_score = get_ml_score(current_txn_df, TRANSACTION_LOG, ML_MODEL)
            except Exception as e:
                print(f"⚠️ ML scoring error: {e}")
                ml_score = 0.5
        else:
            # ⚡ CHANGED: Give neutral score instead of 0.5 when no history
            ml_score = 0.4 if len(TRANSACTION_LOG) < 10 else 0.5

        # 6. Graph-Based Score
        try:
            graph_score, graph_reasons = get_graph_score(txn)
        except Exception as e:
            print(f"⚠️ Graph scoring error: {e}")
            graph_score, graph_reasons = 0.0, []
        
        # 7. AI Innovation Scores
        
        # 7.1. Fake Screenshot Detection - MORE AGGRESSIVE
        screenshot_risk = get_screenshot_risk(txn)
        if screenshot_risk > 0.4:  # Lowered from 0.5
            rule_reasons.append(f"🎯 AI: Fake Screenshot Risk ({screenshot_risk:.2f})")
        
        # 7.2. Voice Scam Detection - MORE AGGRESSIVE
        voice_scam_risk = get_voice_scam_risk(txn)
        if voice_scam_risk > 0.4:  # Lowered from 0.5
            rule_reasons.append(f"📞 AI: Voice Scam Pattern ({voice_scam_risk:.2f})")
            
        # 7.3. Social Engineering Cues - MORE FREQUENT
        message_content = ""
        if random.random() < 0.2:  # Increased from 0.1 (20% chance)
            scam_messages = [
                f"Hi {txn['sender_id']}, urgently click here for your refund of ₹{txn['amount']}. PIN required.",
                f"Your account will be suspended! Verify immediately with OTP.",
                f"Congratulations! You won ₹{txn['amount']}. Claim now with your PIN.",
                f"Urgent refund of ₹{txn['amount']} pending. Confirm your details.",
            ]
            message_content = random.choice(scam_messages)
        
        social_risk, social_reasons = get_social_engineering_score(message_content)
        
        if social_risk > 0.25:  # Lowered from 0.3
            rule_reasons.extend(social_reasons)

        # 8. Final Scoring (Weighted Average + Innovation Contribution)
        final_score = (ML_WEIGHT * ml_score) + (RULE_WEIGHT * rule_score) + (GRAPH_WEIGHT * graph_score)
        
        # Add Innovation Contributions
        final_score += (AI_SCREENSHOT_WEIGHT * screenshot_risk)
        final_score += (AI_VOICE_WEIGHT * voice_scam_risk)
        final_score += (AI_SOCIAL_WEIGHT * social_risk)
        
        # Ensure score remains 0-1
        final_score = np.clip(final_score, 0.0, 1.0)
        
        # 9. Flagging - Using new threshold
        is_high_risk = final_score >= HIGH_RISK_THRESHOLD
        
        # 10. Update In-Memory Log
        log_txn = current_txn_df[['transaction_id', 'timestamp', 'sender_id', 'receiver_id', 'amount']].copy()
        TRANSACTION_LOG = pd.concat([TRANSACTION_LOG, log_txn], ignore_index=True)
        TRANSACTION_LOG = TRANSACTION_LOG.tail(10000)  # Keep last 10K transactions
        
        # 11. Prepare Output
        all_reasons = rule_reasons + graph_reasons
        
        # ⚡ MORE SENSITIVE: Lower threshold for ML anomaly flagging
        if ml_score > 0.6:  # Lowered from 0.7
            all_reasons.append(f"🤖 ML Anomaly Detected (Score: {ml_score:.2f})")
        
        # Add debug info for very high scores
        if final_score > 0.8:
            all_reasons.append(f"⚠️ CRITICAL RISK LEVEL: {final_score:.2f}")
            
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
            'reasons': all_reasons if all_reasons else ["Normal transaction pattern"],
            'user_risk_level': user_risk_level,
            'location': txn['location']
        }
        
        # Log high-risk detections
        if is_high_risk:
            print(f"🚨 HIGH RISK DETECTED: {txn['transaction_id']} | Score: {final_score:.2f} | Amount: ₹{txn['amount']:,.2f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Critical error in fraud detection: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a safe default result
        return {
            'transaction_id': txn.get('transaction_id', 'ERROR'),
            'timestamp': txn.get('timestamp', datetime.now()),
            'sender_id': txn.get('sender_id', 'UNKNOWN'),
            'receiver_id': txn.get('receiver_id', 'UNKNOWN'),
            'amount': txn.get('amount', 0),
            'rule_score': 0.5,
            'ml_score': 0.5,
            'graph_score': 0.0,
            'final_score': 0.5,
            'is_high_risk': False,
            'reasons': [f"Error during fraud detection: {str(e)}"],
            'user_risk_level': 2,
            'location': txn.get('location', 'Unknown')
        }


if __name__ == '__main__':
    print("✅ Fraud Score Service loaded.")
    print(f"⚡ High-Risk Threshold set to: {HIGH_RISK_THRESHOLD}")