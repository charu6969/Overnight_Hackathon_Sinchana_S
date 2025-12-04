import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import os

MODEL_PATH = './models/isolation_forest_model.pkl'
DATA_PATH = './data/clean_transactions.csv'
TIME_WINDOW = '24H' # For velocity features (Used in rolling count below)

def create_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers features required for the Isolation Forest model."""
    
    if df.empty:
        return pd.DataFrame()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. amount (used directly)
    
    # 2. time_of_day (Cyclical feature)
    df['hour'] = df['timestamp'].dt.hour
    df['time_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['time_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Set timestamp as index for rolling window calculations
    df = df.set_index('timestamp').sort_index()

    # 3. transaction_velocity (count in last TIME_WINDOW)
    # This feature works because we are counting 'transaction_id', which is numeric (int) or implicitly counted
    velocity = df.groupby('sender_id').rolling(TIME_WINDOW)['transaction_id'].count().reset_index(level=0, drop=True)
    df['transaction_velocity'] = velocity.fillna(0)
    
    # 4. unique receivers past 24 hrs
    
    # ABSOLUTE WORKAROUND: Avoiding rolling().apply() for string columns that cause ValueError/DataError.
    # This involves grouping and calculating the unique receivers count within a 24-hour window explicitly.
    df['receiver_id'] = df['receiver_id'].astype(str)
    
    # Group by sender and calculate the rolling unique count manually for each group
    unique_receivers_list = []
    
    # Convert 'H' in TIME_WINDOW to timedelta for explicit calculation
    window_timedelta = timedelta(hours=24) 
    
    for sender_id, group in df.groupby('sender_id'):
        unique_count = []
        # Sort group by time (already done via df.sort_index())
        
        # Calculate the unique count for each row explicitly
        for i in range(len(group)):
            current_time = group.index[i]
            # Define the start of the 24-hour window
            start_time = current_time - window_timedelta 
            
            # Filter transactions within the window for the current sender
            window_txns = group.loc[start_time:current_time]
            
            # Count unique receivers in that window
            # Use .iloc[:-1] to exclude the current transaction being processed, mimicking rolling behavior
            unique_receivers_in_window = window_txns['receiver_id'].nunique()
            unique_count.append(unique_receivers_in_window)
            
        # Create a temporary series with the group's index and the calculated counts
        unique_receivers_list.append(pd.Series(unique_count, index=group.index))

    # Combine all sender results into a single Series
    unique_receivers_series = pd.concat(unique_receivers_list)
    
    # Assign the calculated feature back to the main DataFrame
    df['unique_receivers_24h'] = unique_receivers_series.fillna(0)
    
    df = df.reset_index(drop=False) # Put timestamp back as a column
    
    FEATURES = ['amount', 'time_sin', 'time_cos', 'transaction_velocity', 'unique_receivers_24h']
    
    return df[FEATURES]

def train_and_save_model():
    """Loads clean data, trains IsolationForest, and saves the model."""
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Training data not found at {DATA_PATH}. Run simulator.py --generate-clean-data first.")
        return

    print("Loading data and training Isolation Forest...")
    try:
        data = pd.read_csv(DATA_PATH)
    except pd.errors.EmptyDataError:
        print("Error: Clean data file is empty.")
        return
        
    features_df = create_ml_features(data)
    
    if features_df.empty:
        print("Error: Failed to create features for training.")
        return

    # Train Isolation Forest (suitable for unsupervised anomaly detection)
    model = IsolationForest(contamination='auto', random_state=42, n_estimators=100)
    model.fit(features_df)
    
    # Create model directory if it doesn't exist
    if not os.path.exists('./models'):
        os.makedirs('./models')
        
    joblib.dump(model, MODEL_PATH)
    print(f"Isolation Forest model trained and saved to {MODEL_PATH}")

def get_ml_score(current_txn_df: pd.DataFrame, past_txns: pd.DataFrame, model) -> float:
    """Calculates the ML-based anomaly score (0-1)."""
    
    # Combine current transaction with a buffer of past transactions to calculate velocity features
    combined_df = pd.concat([past_txns, current_txn_df], ignore_index=True)
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    
    # Keep log size manageable (e.g., last 10,000 transactions)
    combined_df = combined_df.sort_values('timestamp').tail(10000)

    # Create features for the *entire* combined set
    features_df = create_ml_features(combined_df)

    if features_df.empty:
        return 0.5 

    # The score for the current transaction is the last one in the feature set
    current_txn_features = features_df.tail(1)
    
    # Decision function returns a score where lower is more anomalous (closer to -1)
    raw_score = model.decision_function(current_txn_features)[0]
    
    # Heuristic normalization (higher is riskier)
    ml_score = 1 / (1 + np.exp(raw_score * 20))
    
    return ml_score

if __name__ == '__main__':
    # Initial setup: train the model
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    train_and_save_model()