import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from src.data_preprocessing import load_and_label_data

def visualize_engine_optimized(engine_id, model, scaler, df, sensor_cols, sequence_length):
    engine_df = df[df['id'] == engine_id].copy()
    engine_df['cycle_norm'] = engine_df['cycle']
    engine_df[sensor_cols] = scaler.transform(engine_df[sensor_cols])
    
    # Pre-calculate all sequences for this engine
    engine_data = engine_df[sensor_cols].values
    num_cycles = len(engine_data)
    
    if num_cycles < sequence_length:
        # Handle very short engines with one padded prediction
        padded = np.pad(engine_data, ((sequence_length - num_cycles, 0), (0, 0)), 'constant')
        prob = model.predict(padded.reshape(1, sequence_length, -1), verbose=0)[0][0]
        return [engine_df['cycle'].iloc[-1]], [prob], [engine_df['RUL_actual'].iloc[-1]]

    # BATCH PREDICTION: Create all sequences first
    sequences = []
    for i in range(sequence_length, num_cycles + 1):
        sequences.append(engine_data[i-sequence_length:i])
    
    # Convert to array and predict ALL at once (much faster)
    sequences_array = np.array(sequences).astype(np.float32)
    probs = model.predict(sequences_array, verbose=0).flatten()
    
    # Align results
    cycles = engine_df['cycle'].iloc[sequence_length-1:].values
    ruls = engine_df['RUL_actual'].iloc[sequence_length-1:].values
    
    return cycles, probs, ruls

def main():
    print("Step 1: Loading Data...")
    df = load_and_label_data('data/PM_test.csv')
    truth_df = pd.read_csv('data/PM_truth.csv')
    truth_df.columns = ['id', 'remaining_cycles']
    
    # Data alignment logic
    max_cycle = df.groupby('id')['cycle'].max().reset_index()
    max_cycle.columns = ['id', 'last_cycle']
    truth_df = truth_df.merge(max_cycle, on='id')
    truth_df['true_end'] = truth_df['last_cycle'] + truth_df['remaining_cycles']
    df = df.merge(truth_df[['id', 'true_end']], on='id')
    df['RUL_actual'] = df['true_end'] - df['cycle']
    
    print("Step 2: Loading Model (This might take a second)...")
    scaler = joblib.load('scaler.pkl')
    model = load_model('model.h5')
    sensor_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm'] + \
                  ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']

    # 3. Create Plots
    fig, (ax_h, ax_f) = plt.subplots(2, 1, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.4)

    def format_plot(ax, cycles, probs, ruls, title):
        ax.plot(cycles, probs, color='#d62728', label='Failure Prob', linewidth=2)
        ax.set_ylabel('Probability', color='#d62728', fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color='black', linestyle='--')
        
        ax2 = ax.twinx()
        ax2.plot(cycles, ruls, color='#1f77b4', linestyle=':', label='Actual RUL')
        ax2.set_ylabel('Remaining Cycles', color='#1f77b4', fontweight='bold')
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)

    print("Step 3: Generating Results for Engine 1...")
    c1, p1, r1 = visualize_engine_optimized(1, model, scaler, df, sensor_cols, 50)
    format_plot(ax_h, c1, p1, r1, "Healthy Engine Status (ID: 1)")

    print("Step 4: Generating Results for Engine 34...")
    c34, p34, r34 = visualize_engine_optimized(34, model, scaler, df, sensor_cols, 50)
    format_plot(ax_f, c34, p34, r34, "Failure Approaching Status (ID: 34)")

    plt.savefig('final_report_plot.png')
    print("Success! Plot saved as 'final_report_plot.png'.")
    # Using plt.show() can sometimes freeze VS Code if the GUI backend is buggy.
    # It's safer to just check the saved PNG file.

if __name__ == "__main__":
    main()