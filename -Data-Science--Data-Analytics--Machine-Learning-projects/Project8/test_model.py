import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from src.data_preprocessing import load_and_label_data
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

def test_performance():
    print("Step 1: Aligning Test Data with Ground Truth...")
    df = load_and_label_data('data/PM_test.csv')
    
    # Load Truth with Header support
    truth_df = pd.read_csv('data/PM_truth.csv')
    truth_df.columns = ['id', 'remaining_cycles']
    
    # Align Truth with last recorded cycles
    max_cycle = df.groupby('id')['cycle'].max().reset_index()
    max_cycle.columns = ['id', 'last_cycle_in_file']
    truth_df = truth_df.merge(max_cycle, on='id')
    truth_df['true_end_cycle'] = truth_df['last_cycle_in_file'] + truth_df['remaining_cycles']
    
    df = df.merge(truth_df[['id', 'true_end_cycle']], on='id')
    df['RUL'] = (df['true_end_cycle'] - df['cycle'])
    df['cycle_norm'] = df['cycle']
    df['label_bc'] = np.where(df['RUL'] <= 30, 1, 0)
    
    print("Step 2: Normalizing & Predicting...")
    scaler = joblib.load('scaler.pkl')
    # Match the 18 selected features
    sensor_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm'] + \
                  ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    
    df[sensor_cols] = scaler.transform(df[sensor_cols])
    
    sequence_length = 50
    seq_list = []
    for engine_id in df['id'].unique():
        engine_data = df[df['id'] == engine_id][sensor_cols].values
        if len(engine_data) >= sequence_length:
            seq_list.append(engine_data[-sequence_length:])
        else:
            padded = np.pad(engine_data, ((sequence_length - len(engine_data), 0), (0, 0)), 'constant')
            seq_list.append(padded)
            
    seq_array_test_last = np.asarray(seq_list).astype(np.float32)
    y_true = np.array([df[df['id']==id]['label_bc'].values[-1] for id in df['id'].unique()])

    model = load_model('model.h5')
    y_pred = (model.predict(seq_array_test_last) > 0.5).astype(int).flatten()

    print("\n" + "="*40 + "\n      CLEANED RISK REPORT\n" + "="*40)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2%}")
    print(f"Recall:   {recall_score(y_true, y_pred):.2%}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("="*40)

if __name__ == "__main__":
    test_performance()