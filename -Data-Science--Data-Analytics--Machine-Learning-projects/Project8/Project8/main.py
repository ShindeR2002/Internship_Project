import numpy as np
import joblib
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.data_preprocessing import load_and_label_data, scale_data, gen_sequence, gen_labels
from src.model_training import build_lstm_model

def main():
    print("Step 1: Processing Training Data (Filtering Noisy Sensors)...")
    df = load_and_label_data('data/PM_train.csv')
    df, scaler = scale_data(df)
    joblib.dump(scaler, 'scaler.pkl')

    print("Step 2: Reshaping for LSTM...")
    sequence_length = 50
    # Use only features with variance
    sensor_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm'] + \
                  ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    
    seq_gen = (list(gen_sequence(df[df['id']==id], sequence_length, sensor_cols)) for id in df['id'].unique())
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    
    label_gen = [gen_labels(df[df['id']==id], sequence_length, ['label_bc']) for id in df['id'].unique()]
    label_array = np.concatenate(label_gen).astype(np.float32)

    print("Step 3: Training...")
    model = build_lstm_model(seq_array, label_array)
    
    callbacks = [
        ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, mode='min'),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    ]

    model.fit(seq_array, label_array, epochs=50, batch_size=200, 
              validation_split=0.1, shuffle=True, verbose=1, callbacks=callbacks)
    
    print("Success: Training Complete.")

if __name__ == "__main__":
    main()