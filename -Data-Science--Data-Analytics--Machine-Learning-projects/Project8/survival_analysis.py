import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

def generate_survival_curve(engine_id, model, scaler, df, sensor_cols, seq_len=50):
    # Filter for the engine
    engine_df = df[df['id'] == engine_id].copy()
    engine_df['cycle_norm'] = engine_df['cycle']
    
    # Scale data
    engine_df[sensor_cols] = scaler.transform(engine_df[sensor_cols])
    data_values = engine_df[sensor_cols].values
    
    # FIX: Handle short sequences with Padding
    if len(data_values) < seq_len:
        padding_size = seq_len - len(data_values)
        # Pad with zeros at the beginning
        padded_data = np.pad(data_values, ((padding_size, 0), (0, 0)), mode='constant')
        last_seq = padded_data.reshape(1, seq_len, len(sensor_cols))
    else:
        last_seq = data_values[-seq_len:].reshape(1, seq_len, len(sensor_cols))
    
    # Current risk probability (Hazard)
    current_risk = model.predict(last_seq, verbose=0)[0][0]
    
    # Time horizon for projection (Next 50 cycles)
    time_horizon = np.arange(0, 51)
    
    # Survival probability: P(S) = (1 - risk) ^ (t / scaling_factor)
    # We use a scaling factor of 20 to represent the decay curve
    survival_prob = [(1 - current_risk)**(t / 20) for t in time_horizon]
    
    return time_horizon, survival_prob

def main():
    print("--- Calculating Survival Probabilities (Investment Horizon) ---")
    model = load_model('model.h5')
    scaler = joblib.load('scaler.pkl')
    df = pd.read_csv('data/PM_test.csv') 
    
    sensor_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm'] + \
                  ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']

    plt.figure(figsize=(10, 6))
    
    # Evaluate a Healthy engine (ID 1) and a Critical engine (ID 34)
    for eid, color, label in [(1, '#2ca02c', 'Asset ID 1 (Low Risk Profile)'), 
                               (34, '#d62728', 'Asset ID 34 (High Risk Profile)')]:
        time, prob = generate_survival_curve(eid, model, scaler, df, sensor_cols)
        plt.plot(time, prob, color=color, label=label, linewidth=3)
        
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.3, label='50% Survival Threshold')
    plt.fill_between(np.arange(0, 51), 0, 0.5, color='grey', alpha=0.1, label='Critical Failure Zone')
    
    plt.title("Time-to-Failure Analysis: Future Survival Probability", fontsize=14, pad=15)
    plt.xlabel("Forecast Horizon (Future Cycles / Time Steps)", fontsize=12)
    plt.ylabel("Probability of Continued Operation", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(loc='lower left', frameon=True, shadow=True)
    plt.grid(True, alpha=0.2)
    
    plt.savefig('survival_curves.png')
    print("SUCCESS: Survival analysis complete. View 'survival_curves.png'.")

if __name__ == "__main__":
    main()