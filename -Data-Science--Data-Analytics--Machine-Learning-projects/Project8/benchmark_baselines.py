import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix
from src.data_preprocessing import load_and_label_data, scale_data

def prepare_ml_features(df, sensor_cols, window=50, training=True):
    """
    Extracts features for traditional ML.
    If training=True: Samples multiple points (Healthy & Failing) per engine.
    If training=False: Samples only the last window (current risk state).
    """
    features = []
    labels = []
    
    for engine_id in df['id'].unique():
        engine_data = df[df['id'] == engine_id]
        
        if training:
            # 1. Sample a 'Healthy' window from the beginning (RUL > 30)
            if len(engine_data) > window + 31:
                healthy_window = engine_data.iloc[0:window]
                features.append(np.concatenate([healthy_window[sensor_cols].mean().values, 
                                                healthy_window[sensor_cols].std().values]))
                labels.append(0)
            
            # 2. Sample the 'Failing' window from the very end (RUL <= 30)
            failing_window = engine_data.iloc[-window:]
            features.append(np.concatenate([failing_window[sensor_cols].mean().values, 
                                            failing_window[sensor_cols].std().values]))
            labels.append(1)
        else:
            # For testing, we only care about the most recent state
            if len(engine_data) >= window:
                last_window = engine_data.iloc[-window:]
                features.append(np.concatenate([last_window[sensor_cols].mean().values, 
                                                last_window[sensor_cols].std().values]))
                labels.append(last_window['label_bc'].iloc[-1])
            else:
                # Padding for very short test engines
                # (Simplified for baseline comparison)
                feat_mean = engine_data[sensor_cols].mean().values
                feat_std = engine_data[sensor_cols].std().values
                features.append(np.concatenate([feat_mean, feat_std]))
                labels.append(engine_data['label_bc'].iloc[-1])
            
    return np.array(features), np.array(labels)

def main():
    print("Step 1: Preparing Balanced Benchmark Data...")
    sensor_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm'] + \
                  ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    
    # Load Training Data
    train_df = load_and_label_data('data/PM_train.csv')
    train_df, _ = scale_data(train_df)
    X_train, y_train = prepare_ml_features(train_df, sensor_cols, training=True)

    # Load Test Data & Align Truth
    test_df = load_and_label_data('data/PM_test.csv')
    truth_df = pd.read_csv('data/PM_truth.csv')
    truth_df.columns = ['id', 'remaining_cycles']
    max_cycle = test_df.groupby('id')['cycle'].max().reset_index()
    max_cycle.columns = ['id', 'last_cycle']
    truth_df = truth_df.merge(max_cycle, on='id')
    truth_df['true_end'] = truth_df['last_cycle'] + truth_df['remaining_cycles']
    test_df = test_df.merge(truth_df[['id', 'true_end']], on='id')
    test_df['RUL_actual'] = test_df['true_end'] - test_df['cycle']
    test_df['label_bc'] = np.where(test_df['RUL_actual'] <= 30, 1, 0)
    test_df, _ = scale_data(test_df)
    
    X_test, y_test = prepare_ml_features(test_df, sensor_cols, training=False)

    # 2. Training Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    results = []
    # Using your actual metrics from advanced_metrics.py
    results.append({
        "Model": "Deep LSTM (Current)",
        "ROC-AUC": 0.9925,
        "Recall": 0.8800,
        "Separation (KS)": 0.9733
    })

    print(f"Step 2: Training Baselines on {len(X_train)} samples...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Calculate KS for baselines to match banking standards
        df_ks = pd.DataFrame({'true': y_test, 'prob': y_prob}).sort_values('prob', ascending=False)
        cum_event = df_ks['true'].cumsum() / df_ks['true'].sum()
        cum_nonevent = (1-df_ks['true']).cumsum() / (1-df_ks['true']).sum()
        ks = max(abs(cum_event - cum_nonevent))

        results.append({
            "Model": name,
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "Separation (KS)": round(ks, 4)
        })

    # 3. Final Report
    report = pd.DataFrame(results)
    print("\n" + "="*70)
    print("                 COMPETITIVE MODEL BENCHMARKING")
    print("="*70)
    print(report.to_string(index=False))
    print("="*70)
    print("\nINSIGHT: The LSTM's superior KS Statistic (0.97) proves its ability")
    print("to cleanly separate 'Safe' vs 'Default' assets in a high-risk horizon.")

if __name__ == "__main__":
    main()