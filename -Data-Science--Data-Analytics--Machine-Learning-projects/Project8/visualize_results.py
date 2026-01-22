import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from src.data_preprocessing import load_and_label_data

def visualize_engine(engine_id, model, scaler, df, sensor_cols, sequence_length):
    engine_df = df[df['id'] == engine_id].copy()
    engine_df['cycle_norm'] = engine_df['cycle']
    engine_df[sensor_cols] = scaler.transform(engine_df[sensor_cols])
    
    preds, actual_rul, cycles = [], [], []
    
    if len(engine_df) < sequence_length:
        padded = np.pad(engine_df[sensor_cols].values, ((sequence_length - len(engine_df), 0), (0, 0)), 'constant')
        prob = model.predict(padded.reshape(1, sequence_length, len(sensor_cols)), verbose=0)[0][0]
        return [engine_df['cycle'].iloc[-1]], [prob], [engine_df['RUL_actual'].iloc[-1]]

    for i in range(sequence_length, len(engine_df) + 1):
        seq = engine_df[sensor_cols].values[i-sequence_length:i].reshape(1, sequence_length, len(sensor_cols))
        prob = model.predict(seq, verbose=0)[0][0]
        preds.append(prob)
        actual_rul.append(engine_df['RUL_actual'].iloc[i-1])
        cycles.append(engine_df['cycle'].iloc[i-1])
        
    return cycles, preds, actual_rul

def main():
    # 1. Setup & Data Loading
    df = load_and_label_data('data/PM_test.csv')
    truth_df = pd.read_csv('data/PM_truth.csv')
    truth_df.columns = ['id', 'remaining_cycles']
    
    max_cycle = df.groupby('id')['cycle'].max().reset_index()
    max_cycle.columns = ['id', 'last_cycle']
    truth_df = truth_df.merge(max_cycle, on='id')
    truth_df['true_end'] = truth_df['last_cycle'] + truth_df['remaining_cycles']
    df = df.merge(truth_df[['id', 'true_end']], on='id')
    df['RUL_actual'] = df['true_end'] - df['cycle']
    
    scaler = joblib.load('scaler.pkl')
    model = load_model('model.h5')
    sensor_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm'] + \
                  ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']

    # 2. Plotting Configuration
    fig, (ax_h, ax_f) = plt.subplots(2, 1, figsize=(14, 12))
    plt.subplots_adjust(hspace=0.4) # Add space between subplots

    # Define common plotting logic to ensure consistency
    def format_plot(ax, eid, title_text, cycles, preds, ruls):
        # Failure Probability (Red Line)
        ax.plot(cycles, preds, color='#d62728', label='Model Failure Probability', linewidth=3)
        ax.set_ylabel('Probability (0.0 - 1.0)', color='#d62728', fontsize=12, fontweight='bold')
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Risk Threshold (Action Required)')
        ax.set_ylim(-0.05, 1.05)
        
        # Actual Remaining Useful Life (Blue Dotted Line)
        ax2 = ax.twinx()
        ax2.plot(cycles, ruls, color='#1f77b4', linestyle=':', label='Actual Remaining Useful Life (RUL)', linewidth=2)
        ax2.set_ylabel('Remaining Cycles', color='#1f77b4', fontsize=12, fontweight='bold')
        ax2.axhline(y=30, color='#2ca02c', alpha=0.3, label='Safety Margin (30 Cycles)')
        
        # Labels and Titles
        ax.set_title(f"Aero-Engine Health Monitoring: {title_text}", fontsize=16, pad=15)
        ax.set_xlabel("Operational Cycles (Flight Hours equivalent)", fontsize=12)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, shadow=True)

    # 3. Generate Healthy vs. Failing Comparison
    print("Generating healthy engine plot...")
    c1, p1, r1 = visualize_engine(1, model, scaler, df, sensor_cols, 50)
    format_plot(ax_h, 1, "Engine ID 1 (Healthy/Stable State)", c1, p1, r1)

    print("Generating failing engine plot...")
    c34, p34, r34 = visualize_engine(34, model, scaler, df, sensor_cols, 50)
    format_plot(ax_f, 34, "Engine ID 34 (Degradation Detected)", c34, p34, r34)

    # Save and Show
    output_name = 'maintenance_comparison_v2.png'
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"Success: Comparison saved as '{output_name}'.")
    plt.show()

if __name__ == "__main__":
    main()