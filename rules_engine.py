import pandas as pd
from datetime import datetime, timedelta
import random
import numpy as np

# Import the User Persona Engine to get the user's risk level
from persona_engine import get_user_risk_level

# --- Config ---
GLOBAL_AVG_AMOUNT = 2750  # Estimated mean amount
MAX_TRAVEL_SPEED_KMH = 600  # Maximum believable flight speed

# ⚡ LOWERED THRESHOLD: More sensitive to high-value transactions
HIGH_VALUE_MULTIPLIER = 2.5  # Was 3x, now 2.5x (more sensitive)

# Placeholder coordinates for city simulation (for Impossible Travel)
CITY_COORDS = {
    'Mumbai': (19.0760, 72.8777),
    'Delhi': (28.7041, 77.1025),
    'Bangalore': (12.9716, 77.5946),
    'Chennai': (13.0827, 80.2707),
    'Pune': (18.5204, 73.8567)
}

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the Earth."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def check_impossible_travel(txn: dict, user_past_txns: pd.DataFrame) -> tuple:
    """Checks for Impossible Travel (Device Intelligence 4)."""
    
    if user_past_txns.empty:
        return 0.0, None
    
    user_txns = user_past_txns[user_past_txns['sender_id'] == txn['sender_id']]
    
    if user_txns.empty:
        return 0.0, None
    
    current_time = txn['timestamp']
    current_location = txn['location']
    
    try:
        last_txn = user_txns.sort_values('timestamp', ascending=False).iloc[0]
    except (IndexError, KeyError):
        return 0.0, None
    
    if current_location == last_txn['location']:
        return 0.0, None

    last_time = last_txn['timestamp']
    last_location = last_txn['location']
    
    time_diff_hours = (current_time - last_time).total_seconds() / 3600
    
    if time_diff_hours <= 0:
        return 0.0, None
    
    if last_location in CITY_COORDS and current_location in CITY_COORDS:
        lat1, lon1 = CITY_COORDS[last_location]
        lat2, lon2 = CITY_COORDS[current_location]
        distance_km = haversine_distance(lat1, lon1, lat2, lon2)
        
        speed = distance_km / time_diff_hours
        if speed > MAX_TRAVEL_SPEED_KMH:
            return 0.5, f"🚨 Impossible Travel ({speed:.0f} km/h > {MAX_TRAVEL_SPEED_KMH} km/h)"

    return 0.0, None


def get_rule_score(txn: dict, user_past_txns: pd.DataFrame) -> tuple:
    """Calculates a rule-based fraud score (0-1) and reasons, incorporating innovations."""
    
    rule_score = 0.0
    reasons = []
    
    amount = txn['amount']
    timestamp = txn['timestamp']
    sender_id = txn['sender_id']
    
    # Get User Risk Persona
    user_risk_level = get_user_risk_level(sender_id)
    # ⚡ INCREASED: Higher scrutiny for Risk Level 1 users
    persona_factor = 1.0 + (0.3 if user_risk_level == 1 else 0.1 if user_risk_level == 2 else 0)

    # --- Rule 1: New Device (High Risk) ---
    if txn.get('is_new_device', False):
        score_add = 0.35 * persona_factor  # Increased from 0.3
        rule_score += score_add
        reasons.append(f"🔴 New Device Usage (+{score_add:.2f})")

    # --- Rule 2: Sudden High-Value Transfer ---
    # ⚡ MORE SENSITIVE: Lowered from 3x to 2.5x
    if amount > HIGH_VALUE_MULTIPLIER * GLOBAL_AVG_AMOUNT: 
        score_add = 0.25 * persona_factor  # Increased from 0.2
        rule_score += score_add
        reasons.append(f"💰 High Value Transfer (> ₹{HIGH_VALUE_MULTIPLIER * GLOBAL_AVG_AMOUNT:,.0f}) (+{score_add:.2f})")
    
    # --- Rule 3: Transaction Velocity ---
    # ⚡ MORE SENSITIVE: Detect even 2+ transactions in 10s (was 3+)
    if not user_past_txns.empty:
        recent_txns = user_past_txns[
            (user_past_txns['sender_id'] == sender_id) & 
            (user_past_txns['timestamp'] > timestamp - pd.Timedelta(seconds=10))
        ]
        if len(recent_txns) >= 1:  # Changed from 2 to 1 (more sensitive)
            score_add = 0.3 * persona_factor  # Increased from 0.2
            rule_score += score_add
            reasons.append(f"⚡ High Txn Velocity ({len(recent_txns)+1} txns in <10s) (+{score_add:.2f})")

    # --- Rule 4: Unusual Hour (Night-time) ---
    hour = timestamp.hour
    if hour in range(2, 6):
        score_add = 0.15 * persona_factor  # Increased from 0.1
        rule_score += score_add
        reasons.append(f"🌙 Unusual Hour (Night Time) (+{score_add:.2f})")

    # --- Rule 5: New Receiver ---
    if txn.get('is_new_receiver', False): 
        score_add = 0.15  # Increased from 0.1
        rule_score += score_add
        reasons.append(f"👤 Transaction to New Receiver (+{score_add:.2f})")
        
    # --- Device Intelligence Rules ---
    
    # Impossible Travel Check
    travel_score, travel_reason = check_impossible_travel(txn, user_past_txns)
    if travel_score > 0:
        rule_score += travel_score
        reasons.append(travel_reason)

    # GPS-IP Mismatch (Simulated)
    if txn.get('is_new_device', False) and user_risk_level == 1 and random.random() < 0.4:
        rule_score += 0.25  # Increased from 0.2
        reasons.append("📍 GPS-IP Mismatch (New Device/High Risk User) (+0.25)")

    # Multiple Accounts on Same Device
    device_id = txn.get('device_id', '')
    # ⚡ EXPANDED: More devices flagged as suspicious
    if device_id.startswith('D4') or device_id in ['D001', 'D002', 'D003']:
        rule_score += 0.35  # Increased from 0.3
        reasons.append(f"⚠️ Suspicious Device ({device_id}) (+0.35)")

    # Cap score at 1.0
    rule_score = min(rule_score, 1.0)
    
    return rule_score, reasons


def get_social_engineering_score(message_content: str) -> tuple:
    """LLM/Keyword simulation based on messaging."""
    score = 0.0
    reasons = []
    
    if not message_content:
        return 0.0, []
    
    # ⚡ ENHANCED: More keywords and higher scores
    keywords = {
        'OTP': 0.6,  # Increased
        'PIN': 0.6,  # Increased
        'refund': 0.5,  # Increased
        'urgent': 0.4,  # Increased
        'click here': 0.7,  # Increased
        'verify': 0.45,
        'account': 0.35,
        'suspended': 0.55,
        'prize': 0.5,
        'won': 0.45,
        'claim': 0.5,
        'expire': 0.45,
        'immediately': 0.4,
        'confirm': 0.4
    }
    
    message_lower = message_content.lower()
    
    for kw, val in keywords.items():
        if kw.lower() in message_lower:
            score = max(score, val)
            reasons.append(f"🎯 Social Engineering: '{kw}' detected (+{val})")

    return score, reasons