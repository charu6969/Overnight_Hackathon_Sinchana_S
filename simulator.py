import pandas as pd
import numpy as np
import time
import random
from datetime import datetime, timedelta
import os

# --- Config ---
N_USERS = 1000
N_DEVICES = 500
CITIES = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune']
USER_IDS = [f'U{i:04d}' for i in range(1, N_USERS + 1)]
DEVICE_IDS = [f'D{i:03d}' for i in range(1, N_DEVICES + 1)]
TXN_TYPES = ['P2P', 'P2M']
GLOBAL_AVG_AMOUNT = 2750

# Simple tracking for new device/new receiver/last location logic
USER_PROFILES = {
    user: {
        'devices': {random.choice(DEVICE_IDS)}, 
        'receivers': set(),
        'last_location': random.choice(CITIES) # For Impossible Travel check
    } for user in USER_IDS
}

# Average transaction amount (for rule-based detection)
USER_AVG_AMOUNT = {user: np.random.uniform(1000, 4000) for user in USER_IDS}


def generate_transaction(sender_id, last_txn_time=None, is_fraud=False):
    """Generates a single transaction, optionally injecting fraud."""
    
    timestamp = datetime.now()
    if last_txn_time:
        time_diff = random.uniform(0.1, 1.5)
        timestamp = last_txn_time + timedelta(seconds=time_diff)

    # 1. Base transaction setup
    sender_profile = USER_PROFILES[sender_id]
    
    receiver_id = random.choice(USER_IDS)
    while receiver_id == sender_id:
        receiver_id = random.choice(USER_IDS)
        
    amount = np.clip(np.random.normal(USER_AVG_AMOUNT[sender_id], USER_AVG_AMOUNT[sender_id] * 0.5), 10, 50000)
    
    # Use last known device/location for consistency
    device_id = random.choice(list(sender_profile['devices']))
    location = sender_profile['last_location']
    txn_type = random.choice(TXN_TYPES)

    # --- Fraud Injection Logic ---
    is_new_device, is_new_receiver, is_high_value, is_night_time = False, False, False, False
    
    if is_fraud or random.random() < 0.05: # 5% baseline fraud chance
        fraud_type = random.choice(['high_value', 'velocity', 'new_device', 'new_receiver', 'night_time', 'impossible_travel', 'mule_account'])
        
        if fraud_type == 'high_value':
            amount = USER_AVG_AMOUNT[sender_id] * np.random.uniform(3.5, 8.0)
            is_high_value = True
        
        elif fraud_type == 'new_device':
            new_device = random.choice(DEVICE_IDS)
            if new_device not in sender_profile['devices']:
                device_id = new_device
                is_new_device = True
                
        elif fraud_type == 'new_receiver':
            new_receiver = f'NEW_U{random.randint(2000, 3000)}'
            if new_receiver not in sender_profile['receivers']:
                receiver_id = new_receiver
                is_new_receiver = True
        
        elif fraud_type == 'night_time':
            # Set time to 2 AM to 5 AM
            timestamp = timestamp.replace(hour=random.randint(2, 5), minute=random.randint(0, 59), second=random.randint(0, 59))
            is_night_time = True
        
        elif fraud_type == 'mule_account':
            # High amount to a known receiver (mule-account pattern)
            if sender_profile['receivers']:
                 receiver_id = random.choice(list(sender_profile['receivers']))
                 amount = USER_AVG_AMOUNT[sender_id] * np.random.uniform(2.5, 4.0)

        elif fraud_type == 'impossible_travel':
            # Simulate travel from Delhi to Chennai in 5 minutes (for Rule 4)
            if location == 'Delhi':
                 location = 'Chennai'
                 # Ensure time difference is small
                 timestamp = last_txn_time + timedelta(minutes=random.uniform(2, 5))
            elif location == 'Mumbai':
                 location = 'Bangalore'
                 timestamp = last_txn_time + timedelta(minutes=random.uniform(2, 5))
    
    # Update profiles for next transaction simulation
    if device_id not in sender_profile['devices']:
         sender_profile['devices'].add(device_id)
         
    if receiver_id not in sender_profile['receivers']:
         sender_profile['receivers'].add(receiver_id)
         
    sender_profile['last_location'] = location


    txn = {
        'transaction_id': f'T{int(time.time() * 100000)}_{random.randint(100, 999)}',
        'timestamp': timestamp,
        'sender_id': sender_id,
        'receiver_id': receiver_id,
        'amount': round(amount, 2),
        'device_id': device_id,
        'location': location,
        'transaction_type': txn_type,
        'is_new_device': is_new_device,
        'is_new_receiver': is_new_receiver,
        'is_high_value_simulated': is_high_value,
        'is_night_time_simulated': is_night_time,
    }
    return txn

def generate_clean_data(n_txns=20000):
    """Generates a large dataset of 'clean' transactions for ML training."""
    print(f"Generating {n_txns} clean transactions for training...")
    clean_data = []
    
    # Reset profiles locally for clean generation
    local_user_profiles = {user: {'devices': {random.choice(DEVICE_IDS)}, 'receivers': set(), 'last_location': random.choice(CITIES)} for user in USER_IDS}
    
    for i in range(n_txns):
        sender = random.choice(USER_IDS)
        receiver = random.choice(USER_IDS)
        amount = np.clip(np.random.normal(USER_AVG_AMOUNT[sender], USER_AVG_AMOUNT[sender] * 0.4), 10, 20000)
        
        txn = {
            'transaction_id': f'T_CLEAN_{i}',
            'timestamp': datetime.now() - timedelta(minutes=random.randint(1, 1000)),
            'sender_id': sender,
            'receiver_id': receiver,
            'amount': round(amount, 2),
            'device_id': random.choice(list(local_user_profiles[sender]['devices'])),
            'location': random.choice(CITIES),
            'transaction_type': random.choice(TXN_TYPES),
            'is_new_device': False,
            'is_new_receiver': False,
            'is_high_value_simulated': False,
            'is_night_time_simulated': False,
        }
        clean_data.append(txn)
    
    df = pd.DataFrame(clean_data)
    df.to_csv('./data/clean_transactions.csv', index=False)
    print("Clean data saved to ./data/clean_transactions.csv")


def run_simulator(transaction_queue):
    """Continuously generates transactions and puts them into a queue."""
    print("Starting real-time transaction simulator...")
    last_txn_time = datetime.now()
    
    while True:
        sender_id = random.choice(USER_IDS)
        
        # Inject velocity fraud every ~20 transactions
        if random.random() < 0.05:
            # Multiple transfers in short time (Velocity Fraud)
            num_burst_txns = random.randint(3, 5)
            # print(f"--- Injecting {num_burst_txns} burst txns (Velocity Fraud) for {sender_id} ---")
            for _ in range(num_burst_txns):
                txn = generate_transaction(sender_id, last_txn_time, is_fraud=True)
                # FIX: Use .put() for queue objects
                transaction_queue.put(txn) 
                last_txn_time = txn['timestamp']
                time.sleep(0.05) 
        
        # Single transaction
        else:
            txn = generate_transaction(sender_id, last_txn_time, is_fraud=False)
            # FIX: Use .put() for queue objects
            transaction_queue.put(txn)
            last_txn_time = txn['timestamp']
            
        time.sleep(random.uniform(0.5, 1.5)) # Standard transaction interval

if __name__ == '__main__':
    import sys
    if '--generate-clean-data' in sys.argv:
        # Create data folder if it doesn't exist
        os.makedirs('./data', exist_ok=True)