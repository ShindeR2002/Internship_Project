import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.data_preprocessing import load_and_label_data
from sklearn.metrics import confusion_matrix

def main():
    print("--- STARTING MODEL STRESS TEST (SENSOR NOISE INJECTION) ---")
    
    # 1. Load Model and Data
    model = load_model('model.h5')
    scaler = joblib.load('scaler.pkl')
    df = load_and_label_data('data/PM_test.csv')
    
    # Setup for sequence prediction (reusing our existing logic)
    sensor_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm'] + \
                  ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    df['cycle_norm'] = df['cycle']
    
    noise_levels = [0, 0.1, 0.5, 1.0, 2.0] # Standard deviations of noise
    results = []

    for noise in noise_levels:
        test_df = df.copy()
        
        # Inject noise specifically into the lead indicator (Sensor 11)
        if noise > 0:
            test_df['s11'] += np.random.normal(0, noise, size=len(test_df))
        
        # Scale and prepare sequences
        test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])
        
        # Prediction Logic (Simplified for stress test)
        # (Assuming we evaluate the latest state of each engine)
        y_true, y_prob = [], []
        for eid in test_df['id'].unique():
            engine_data = test_df[test_df['id'] == eid]
            if len(engine_data) >= 50:
                seq = engine_data[sensor_cols].values[-50:].reshape(1, 50, 18)
                y_prob.append(model.predict(seq, verbose=0)[0][0])
                # Determine ground truth (logic from our metrics script)
                y_true.append(1 if engine_data['cycle'].max() > 150 else 0) # Proxy for failure

        # Calculate Cost Savings for this noise level
        y_pred = (np.array(y_prob) > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        savings = ((tp + fn) * 10000) - ((tp + fp) * 500 + fn * 10000)
        
        results.append({'Noise_Level': noise, 'Savings': savings})

    # 2. Results Visualization
    res_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 5))
    plt.plot(res_df['Noise_Level'], res_df['Savings'], marker='o', linestyle='--', color='darkred')
    plt.title("Model Robustness: ROI vs. Sensor Data Quality", fontsize=14)
    plt.xlabel("Noise Intensity (Sigma) injected into Sensor 11", fontsize=12)
    plt.ylabel("Economic Savings (USD)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('stress_test_results.png')
    
    print("\nSTRESS TEST COMPLETE")
    print("="*40)
    print(res_df.to_string(index=False))
    print("="*40)

if __name__ == "__main__":
    main()