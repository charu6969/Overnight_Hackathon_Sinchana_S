# ai_fraud_module.py (Simplified Logic for Hackathon)
import random

def get_screenshot_risk(txn: dict) -> float:
    """
    (1) Simulates AI-based fake screenshot detection risk.
    High risk if amount > 50k AND user is L1/L2 AND time is unusual.
    """
    is_high_amount = txn['amount'] > 50000
    is_low_risk_user = txn.get('user_risk_level') in [1, 2] # Senior/Rural
    
    # Inject high risk 10% of the time on high-amount transactions
    if is_high_amount and is_low_risk_user and random.random() < 0.1:
        return 0.9 + random.random() * 0.1
    return 0.1 + random.random() * 0.2

def get_voice_scam_risk(txn: dict) -> float:
    """
    (2) Simulates AI call risk analysis.
    Assume a risk score is provided by a pre-analysis module (0-1).
    """
    # Simulate a high voice scam score linked to a high-risk transaction
    if txn.get('is_high_value_simulated', False) and txn.get('is_night_time_simulated', False):
        return 0.8 + random.random() * 0.2
    return 0.05 + random.random() * 0.15

# (3) Social Engineering Behavioral Cues - Simple Keyword Analysis
def get_social_engineering_risk(message_content: str) -> float:
    """LLM simulation based on keywords."""
    score = 0.0
    reasons = []
    
    keywords = {'refund': 0.3, 'OTP': 0.4, 'PIN': 0.5, 'urgent': 0.2, 'click here': 0.6}
    
    for kw, val in keywords.items():
        if kw in message_content.lower():
            score = max(score, val)
            reasons.append(f"Keyword '{kw}' detected (+{val})")

    return score, reasons