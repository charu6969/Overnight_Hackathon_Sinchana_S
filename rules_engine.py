import pandas as pd
from datetime import datetime, timedelta
import random
import numpy as np

# Import the User Persona Engine to get the user's risk level
from persona_engine import get_user_risk_level

# --- Config ---
GLOBAL_AVG_AMOUNT = 2750 # Estimated mean amount
MAX_TRAVEL_SPEED_KMH = 600 # Maximum believable flight speed
DISTANCE_DELHI_CHENNAI_KM = 2100 # Approx. distance

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
    
    current_time = txn['timestamp']
    current_location = txn['location']
    
    # Get the last transaction for the sender
    last_txn = user_past_txns[user_past_txns['sender_id'] == txn['sender_id']].sort_values('timestamp', ascending=False).iloc[0] if not user_past_txns.empty else None
    
    if last_txn is None or current_location == last_txn['location']:
        return 0.0, None

    last_time = last_txn['timestamp']
    last_location = last_txn['location']
    
    time_diff_hours = (current_time - last_time).total_seconds() / 3600
    
    # Calculate distance (simplified check using CITY_COORDS)
    if last_location in CITY_COORDS and current_location in CITY_COORDS:
        lat1, lon1 = CITY_COORDS[last_location]
        lat2, lon2 = CITY_COORDS[current_location]
        distance_km = haversine_distance(lat1, lon1, lat2, lon2)
        
        if time_diff_hours > 0:
            speed = distance_km / time_diff_hours
            if speed > MAX_TRAVEL_SPEED_KMH:
                return 0.4, f"Impossible Travel ({speed:.0f} km/h > {MAX_TRAVEL_SPEED_KMH} km/h)"

    return 0.0, None


def get_rule_score(txn: dict, user_past_txns: pd.DataFrame) -> tuple:
    """Calculates a rule-based fraud score (0-1) and reasons, incorporating innovations."""
    
    rule_score = 0.0
    reasons = []
    
    amount = txn['amount']
    timestamp = txn['timestamp']
    sender_id = txn['sender_id']
    
    # --- Innovation (6): Get User Risk Persona ---
    user_risk_level = get_user_risk_level(sender_id)
    # Persona Factor: Scrutiny increases for Risk Level 1 (Senior/New)
    persona_factor = 1.0 + (0.2 if user_risk_level == 1 else 0)

    # --- Rule 1: New Device (High Risk) ---
    if txn.get('is_new_device', False):
        score_add = 0.3 * persona_factor
        rule_score += score_add
        reasons.append(f"New Device Usage (+{score_add:.2f})")

    # --- Rule 2: Sudden High-Value Transfer ---
    if amount > 3 * GLOBAL_AVG_AMOUNT: 
        score_add = 0.2 * persona_factor
        rule_score += score_add
        reasons.append(f"High Value Transfer (> {3 * GLOBAL_AVG_AMOUNT:,.0f}) (+{score_add:.2f})")
    
    # --- Rule 3: Transaction Velocity ---
    recent_txns = user_past_txns[
        (user_past_txns['sender_id'] == sender_id) & 
        (user_past_txns['timestamp'] > timestamp - pd.Timedelta(seconds=10))
    ]
    if len(recent_txns) >= 2:
        score_add = 0.2 * persona_factor
        rule_score += score_add
        reasons.append(f"High Txn Velocity ({len(recent_txns)+1} txns in <10s) (+{score_add:.2f})")

    # --- Rule 4: Unusual Hour (Night-time) ---
    hour = timestamp.hour
    if hour in range(2, 6):
        score_add = 0.1 * persona_factor
        rule_score += score_add
        reasons.append(f"Unusual Hour (Night Time) (+{score_add:.2f})")

    # --- Rule 5: New Receiver ---
    if txn.get('is_new_receiver', False): 
        score_add = 0.1
        rule_score += score_add
        reasons.append(f"Transaction to New Receiver (+{score_add:.2f})")
        
    # --- Innovation (4): Device Intelligence Rules ---
    
    # 5.1 Impossible Travel Check
    if not user_past_txns.empty:
        travel_score, travel_reason = check_impossible_travel(txn, user_past_txns)
        if travel_score > 0:
            rule_score += travel_score
            reasons.append(travel_reason + f" (+{travel_score:.1f})")

    # 5.2 GPS-IP Mismatch (Simulated)
    # Assume mismatch risk is simulated based on user persona and new device status
    if txn.get('is_new_device', False) and user_risk_level == 1 and random.random() < 0.3:
        rule_score += 0.2
        reasons.append("GPS-IP Mismatch (New Device/L1 User) (+0.2)")

    # 5.3 Multiple Accounts on Same Device (Simulated)
    if txn.get('device_id') == 'D001' or txn.get('device_id') == 'D002': # Assume D001/D002 are known fraud devices
        rule_score += 0.3
        reasons.append("Known Fraud Device Usage (+0.3)")


    # Cap score at 1.0
    rule_score = min(rule_score, 1.0)
    
    return rule_score, reasons

# --- Innovation (3): Social Engineering Rule (Used in Fraud Score Service) ---
def get_social_engineering_score(message_content: str) -> tuple:
    """LLM/Keyword simulation based on messaging."""
    score = 0.0
    reasons = []
    
    keywords = {'OTP': 0.5, 'PIN': 0.5, 'refund': 0.4, 'urgent': 0.3, 'click here': 0.6}
    
    # Randomly inject a message if the transaction is flagged
    if random.random() < 0.2:
        random_keyword = random.choice(list(keywords.keys()))
        message_content = f"Your account needs urgent {random_keyword} verification."

    for kw, val in keywords.items():
        if kw in message_content.lower():
            score = max(score, val)
            reasons.append(f"Social Engineering Keyword '{kw}' detected (+{val})")

    return score, reasons