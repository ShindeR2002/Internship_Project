import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from src.data_preprocessing import load_and_label_data
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix

def calculate_ks_statistic(y_true, y_prob):
    df = pd.DataFrame({'true': y_true, 'prob': y_prob})
    df = df.sort_values(by='prob', ascending=False)
    df['cum_event'] = df['true'].cumsum() / df['true'].sum()
    df['cum_non_event'] = (1 - df['true']).cumsum() / (1 - df['true']).sum()
    return max(abs(df['cum_event'] - df['cum_non_event']))

def main():
    # 1. Load Data & Model
    df = load_and_label_data('data/PM_test.csv')
    truth_df = pd.read_csv('data/PM_truth.csv')
    truth_df.columns = ['id', 'remaining_cycles']
    
    # Process RUL and Labels
    max_cycle = df.groupby('id')['cycle'].max().reset_index()
    max_cycle.columns = ['id', 'last_cycle']
    truth_df = truth_df.merge(max_cycle, on='id')
    truth_df['true_end'] = truth_df['last_cycle'] + truth_df['remaining_cycles']
    df = df.merge(truth_df[['id', 'true_end']], on='id')
    df['RUL_actual'] = df['true_end'] - df['cycle']
    df['label_bc'] = np.where(df['RUL_actual'] <= 30, 1, 0)

    # 2. Scaling & Prediction Prep
    scaler = joblib.load('scaler.pkl')
    sensor_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm'] + \
                  ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    df['cycle_norm'] = df['cycle']
    df[sensor_cols] = scaler.transform(df[sensor_cols])

    sequence_length = 50
    seq_list, y_true_list = [], []
    for engine_id in df['id'].unique():
        engine_data = df[df['id'] == engine_id][sensor_cols].values
        if len(engine_data) >= sequence_length:
            seq_list.append(engine_data[-sequence_length:])
        else:
            padded = np.pad(engine_data, ((sequence_length - len(engine_data), 0), (0, 0)), 'constant')
            seq_list.append(padded)
        y_true_list.append(df[df['id'] == engine_id]['label_bc'].values[-1])
            
    seq_array = np.asarray(seq_list).astype(np.float32)
    y_true = np.array(y_true_list) # Define y_true here

    model = load_model('model.h5')
    y_prob = model.predict(seq_array, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    # 3. Business Impact Logic
    cost_failure = 10000 
    cost_preventive = 500
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Reactive: Fix only when they break (TP + FN)
    cost_reactive = (tp + fn) * cost_failure 
    # Predictive: Cost of inspections (TP + FP) + Cost of missed failures (FN)
    cost_model = ((tp + fp) * cost_preventive) + (fn * cost_failure)
    
    savings = cost_reactive - cost_model
    savings_pct = (savings / cost_reactive) * 100 if cost_reactive > 0 else 0

    # 4. Final Professional Reporting
    print("\n" + "="*45)
    print("   FINANCIAL RISK & IMPACT REPORT")
    print("-" * 45)
    print(f"ROC-AUC Score:      {roc_auc_score(y_true, y_prob):.4f}")
    print(f"KS Statistic:       {calculate_ks_statistic(y_true, y_prob):.4f}")
    print(f"Failures Caught:    {tp}/{tp+fn}")
    print(f"False Alarms:       {fp}")
    print("\n" + "-" * 45)
    print("   ECONOMIC SIMULATION (USD)")
    print("-" * 45)
    print(f"Cost of Unscheduled Failure:  ${cost_failure:,.0f}")
    print(f"Cost of Planned Maintenance:  ${cost_preventive:,.0f}")
    print(f"Reactive Baseline Cost:       ${cost_reactive:,.0f}")
    print(f"LSTM Strategy Cost:           ${cost_model:,.0f}")
    print(f"NET ECONOMIC SAVINGS:         ${savings:,.0f}")
    print(f"TOTAL COST REDUCTION:         {savings_pct:.1f}%")
    print("="*45)

if __name__ == "__main__":
    main()