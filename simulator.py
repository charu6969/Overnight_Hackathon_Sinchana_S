import pandas as pd
from datetime import datetime, timedelta
import random
import numpy as np
import time
import os

# --- Config ---
N_USERS = 1000
N_DEVICES = 500
CITIES = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune']
USER_IDS = [f'U{i:04d}' for i in range(1, N_USERS + 1)]
DEVICE_IDS = [f'D{i:03d}' for i in range(1, N_DEVICES + 1)]
TXN_TYPES = ['P2P', 'P2M']
GLOBAL_AVG_AMOUNT = 2750

# ⚡ INCREASED FRAUD RATE - 25% instead of 5%
FRAUD_PROBABILITY = 0.25  # 25% of transactions will be fraudulent

# Simple tracking for new device/new receiver/last location logic
USER_PROFILES = {
    user: {
        'devices': {random.choice(DEVICE_IDS)}, 
        'receivers': set(),
        'last_location': random.choice(CITIES),
        'last_txn_time': None  # Track for velocity detection
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
    device_id = random.choice(list(sender_profile['devices'])) if sender_profile['devices'] else random.choice(DEVICE_IDS)
    location = sender_profile['last_location']
    txn_type = random.choice(TXN_TYPES)

    # --- Enhanced Fraud Injection Logic ---
    is_new_device, is_new_receiver, is_high_value, is_night_time = False, False, False, False
    
    # ⚡ INCREASED: Now 25% baseline fraud chance (was 5%)
    if is_fraud or random.random() < FRAUD_PROBABILITY:
        # Multiple fraud types can be combined for higher risk
        fraud_types = []
        
        # Randomly select 1-3 fraud patterns
        num_fraud_patterns = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
        fraud_types = random.sample([
            'high_value', 'new_device', 'new_receiver', 
            'night_time', 'impossible_travel', 'velocity', 'mule_account'
        ], num_fraud_patterns)
        
        for fraud_type in fraud_types:
            if fraud_type == 'high_value':
                # ⚡ Much higher amounts to trigger detection
                amount = USER_AVG_AMOUNT[sender_id] * np.random.uniform(4.0, 10.0)
                is_high_value = True
            
            elif fraud_type == 'new_device':
                # Always use a new device for fraud
                new_device = f'D{random.randint(400, 499):03d}'  # Use high device IDs for fraud
                device_id = new_device
                is_new_device = True
                    
            elif fraud_type == 'new_receiver':
                # Send to completely new receiver
                new_receiver = f'NEW_U{random.randint(2000, 5000)}'
                receiver_id = new_receiver
                is_new_receiver = True
            
            elif fraud_type == 'night_time':
                # Set time to suspicious hours (2 AM - 5 AM)
                timestamp = timestamp.replace(
                    hour=random.randint(2, 5), 
                    minute=random.randint(0, 59), 
                    second=random.randint(0, 59)
                )
                is_night_time = True
            
            elif fraud_type == 'mule_account':
                # High amount to a known receiver (mule-account pattern)
                if sender_profile['receivers']:
                    receiver_id = random.choice(list(sender_profile['receivers']))
                    amount = USER_AVG_AMOUNT[sender_id] * np.random.uniform(3.0, 6.0)
                else:
                    # Create a mule account relationship
                    receiver_id = f'MULE_U{random.randint(6000, 7000)}'
                    amount = USER_AVG_AMOUNT[sender_id] * np.random.uniform(3.0, 6.0)

            elif fraud_type == 'impossible_travel':
                # Simulate impossible travel - far cities in short time
                travel_pairs = [
                    ('Delhi', 'Chennai'),
                    ('Mumbai', 'Bangalore'),
                    ('Pune', 'Delhi'),
                    ('Chennai', 'Mumbai')
                ]
                from_city, to_city = random.choice(travel_pairs)
                
                if sender_profile['last_location'] == from_city:
                    location = to_city
                    # Very short time between transactions
                    if sender_profile['last_txn_time']:
                        timestamp = sender_profile['last_txn_time'] + timedelta(minutes=random.uniform(1, 3))
                else:
                    # Just move to a far city
                    location = random.choice([c for c in CITIES if c != sender_profile['last_location']])
                    if sender_profile['last_txn_time']:
                        timestamp = sender_profile['last_txn_time'] + timedelta(minutes=random.uniform(1, 4))
            
            elif fraud_type == 'velocity':
                # This is handled in run_simulator with burst transactions
                pass
    
    # Update profiles for next transaction simulation
    if device_id not in sender_profile['devices']:
        sender_profile['devices'].add(device_id)
         
    if receiver_id not in sender_profile['receivers']:
        sender_profile['receivers'].add(receiver_id)
         
    sender_profile['last_location'] = location
    sender_profile['last_txn_time'] = timestamp

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
    print(f"📊 Generating {n_txns} clean transactions for training...")
    clean_data = []
    
    # Reset profiles locally for clean generation
    local_user_profiles = {
        user: {
            'devices': {random.choice(DEVICE_IDS)}, 
            'receivers': set(), 
            'last_location': random.choice(CITIES)
        } for user in USER_IDS
    }
    
    for i in range(n_txns):
        sender = random.choice(USER_IDS)
        receiver = random.choice(USER_IDS)
        while receiver == sender:
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
        
        if (i + 1) % 5000 == 0:
            print(f"  Generated {i + 1}/{n_txns} transactions...")
    
    df = pd.DataFrame(clean_data)
    
    # Ensure data directory exists
    os.makedirs('./data', exist_ok=True)
    df.to_csv('./data/clean_transactions.csv', index=False)
    print("✅ Clean data saved to ./data/clean_transactions.csv")


def run_simulator(transaction_queue):
    """Continuously generates transactions and puts them into a queue."""
    print("🚀 Starting real-time transaction simulator...")
    print("⚡ Enhanced Fraud Detection Mode - 25% fraud rate")
    print("=" * 60)
    
    last_txn_time = datetime.now()
    txn_count = 0
    fraud_count = 0
    
    try:
        while True:
            sender_id = random.choice(USER_IDS)
            
            # ⚡ INCREASED: Inject velocity fraud more frequently (15% chance, was 5%)
            if random.random() < 0.15:
                # Multiple transfers in short time (Velocity Fraud)
                num_burst_txns = random.randint(3, 7)  # Increased burst size
                print(f"⚡ FRAUD INJECTION: {num_burst_txns} burst txns (Velocity) for {sender_id}")
                
                for _ in range(num_burst_txns):
                    txn = generate_transaction(sender_id, last_txn_time, is_fraud=True)
                    transaction_queue.put(txn)
                    last_txn_time = txn['timestamp']
                    txn_count += 1
                    fraud_count += 1
                    time.sleep(0.02)  # Very short delay for velocity detection
            else:
                # Single transaction (may or may not be fraud based on FRAUD_PROBABILITY)
                is_fraud_txn = random.random() < FRAUD_PROBABILITY
                txn = generate_transaction(sender_id, last_txn_time, is_fraud=is_fraud_txn)
                transaction_queue.put(txn)
                last_txn_time = txn['timestamp']
                txn_count += 1
                
                if is_fraud_txn:
                    fraud_count += 1
                
                # Log progress every 25 transactions (more frequent updates)
                if txn_count % 25 == 0:
                    fraud_rate = (fraud_count / txn_count) * 100
                    print(f"📊 Processed {txn_count} txns | Fraud: {fraud_count} ({fraud_rate:.1f}%)")
            
            # Shorter delay for faster transaction generation
            time.sleep(random.uniform(0.3, 1.0))
            
    except KeyboardInterrupt:
        fraud_rate = (fraud_count / txn_count) * 100 if txn_count > 0 else 0
        print(f"\n🛑 Simulator stopped.")
        print(f"📊 Final Stats: {txn_count} transactions | {fraud_count} fraudulent ({fraud_rate:.1f}%)")
    except Exception as e:
        print(f"❌ Simulator error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import sys
    if '--generate-clean-data' in sys.argv:
        # Create data folder if it doesn't exist
        os.makedirs('./data', exist_ok=True)
        generate_clean_data()
    else:
        print("Usage:")
        print("  python simulator.py --generate-clean-data  # Generate training data")
        print("\nNote: Simulator is normally run from dashboard.py")