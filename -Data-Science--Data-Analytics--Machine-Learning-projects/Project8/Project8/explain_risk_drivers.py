import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from src.data_preprocessing import load_and_label_data, scale_data

def main():
    print("Step 1: Extracting Feature Importance from Baseline...")
    sensor_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm'] + \
                  ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    
    # Load and process data
    train_df = load_and_label_data('data/PM_train.csv')
    train_df, _ = scale_data(train_df)
    
    # Prepare features (Healthy vs Failing)
    features, labels = [], []
    for engine_id in train_df['id'].unique():
        engine_data = train_df[train_df['id'] == engine_id]
        # Sample Healthy
        if len(engine_data) > 60:
            features.append(engine_data.iloc[0:50][sensor_cols].mean().values)
            labels.append(0)
        # Sample Failing
        features.append(engine_data.iloc[-50:][sensor_cols].mean().values)
        labels.append(1)

    # Train Random Forest to get importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(features, labels)
    
    # Map back to sensor names
    importances = pd.DataFrame({
        'Feature': sensor_cols,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # 3. Visualization
    plt.figure(figsize=(10, 6))
    plt.barh(importances['Feature'][:10], importances['Importance'][:10], color='teal')
    plt.gca().invert_yaxis()
    plt.title("Risk Driver Analysis: Top 10 Indicators of Asset Failure", fontsize=14)
    plt.xlabel("Predictive Power (Importance Score)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.savefig('risk_drivers.png')
    print("\n" + "="*45)
    print("   TOP 5 RISK INDICATORS (LEAD SIGNALS)")
    print("="*45)
    print(importances.head(5).to_string(index=False))
    print("="*45)
    print("\nSuccess: Analysis saved as 'risk_drivers.png'.")

if __name__ == "__main__":
    main()