import random
import pandas as pd
import os 

# Assume a small CSV mapping User IDs to a fixed persona risk level
PERSONA_DATA_PATH = './data/user_personas.csv'

def setup_user_personas(user_ids):
    """Initializes a simple persona profile for all users."""
    if not os.path.exists('./data'): os.makedirs('./data')
    
    # Risk Level 1 (Senior/Rural) - Highest scrutiny
    # Risk Level 2 (Moderate) - Default scrutiny
    # Risk Level 3 (Expert/High Txn) - Lowest scrutiny
    
    data = {'sender_id': user_ids, 
            'risk_level': [random.choice([1, 1, 2, 2, 2, 3]) for _ in user_ids]}
    df = pd.DataFrame(data)
    df.to_csv(PERSONA_DATA_PATH, index=False)
    return df

def get_user_risk_level(user_id: str) -> int:
    """Looks up the pre-defined risk level for a user."""
    try:
        df = pd.read_csv(PERSONA_DATA_PATH)
        level = df[df['sender_id'] == user_id]['risk_level'].iloc[0]
        return int(level)
    except Exception:
        return 2 # Default to Moderate if not found