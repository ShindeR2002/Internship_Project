import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from src.data_preprocessing import load_and_label_data

def main():
    # 1. Load Data & Model
    df = load_and_label_data('data/PM_test.csv')
    model = load_model('model.h5')
    scaler = joblib.load('scaler.pkl')
    
    # Preprocessing for predictions
    sensor_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm'] + \
                  ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    df['cycle_norm'] = df['cycle']
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    relevant_cols = [col for col in df.columns if 's' in col or 'label' in col]
    sns.heatmap(df[relevant_cols].corr(), annot=False, cmap='RdYlGn', center=0)
    plt.title("Sensor Correlation Map: Identifying Failure Patterns", fontsize=15)
    plt.tight_layout()
    plt.savefig('sensor_heatmap.png')
    print("Saved: sensor_heatmap.png")

    # 3. Leading Indicator Trend (Path to Failure)
    plt.figure(figsize=(10, 6))
    # Pick specific engine IDs to show different lifecycle stages
    for eid in [1, 15, 34]: 
        subset = df[df['id'] == eid]
        plt.plot(subset['cycle'], subset['s11'], label=f'Engine ID {eid}', linewidth=2)
    
    plt.title("Sensor 11: The 'Golden Signal' for Failure Prediction", fontsize=14)
    plt.xlabel("Operational Cycles")
    plt.ylabel("Raw Sensor Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('leading_indicator.png')
    print("Saved: leading_indicator.png")

    # 4. Model Confidence Distribution
    # Prepare sequences for all data points to see the probability distribution
    sequence_length = 50
    seq_list = []
    for engine_id in df['id'].unique():
        engine_data = df[df['id'] == engine_id][sensor_cols].values
        # Apply scaling inside the loop for the visualization
        engine_data_scaled = scaler.transform(engine_data)
        if len(engine_data_scaled) >= sequence_length:
            seq_list.append(engine_data_scaled[-sequence_length:])
            
    seq_array = np.asarray(seq_list).astype(np.float32)
    y_prob = model.predict(seq_array, verbose=0).flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(y_prob, bins=25, color='#3498db', edgecolor='white', alpha=0.8)
    plt.title("Model Confidence Distribution (Risk Probabilities)", fontsize=14)
    plt.xlabel("Predicted Probability of Failure (0 = Healthy, 1 = Terminal)")
    plt.ylabel("Number of Assets")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('confidence_histogram.png')
    print("Saved: confidence_histogram.png")

if __name__ == "__main__":
    main()