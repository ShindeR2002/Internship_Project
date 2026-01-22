import pandas as pd
import numpy as np
from sklearn import preprocessing

def load_and_label_data(train_path):
    # Load with header as your files contain them
    df = pd.read_csv(train_path)
    
    # Ensure IDs and cycles are sorted
    df = df.sort_values(['id', 'cycle'])
    
    # Calculate RUL (Remaining Useful Life)
    max_cycle = df.groupby('id')['cycle'].max().reset_index()
    max_cycle.columns = ['id', 'max']
    df = df.merge(max_cycle, on=['id'], how='left')
    df['RUL'] = df['max'] - df['cycle']
    
    # ADVANCEMENT: RUL Clipping at 125
    # This prevents 'Healthy' data from drowning out the 'Failure' signals.
    df['RUL_clipped'] = df['RUL'].clip(upper=125)
    
    # Label: 1 if failure is within 30 cycles
    df['label_bc'] = np.where(df['RUL'] <= 30, 1, 0)
    df.drop('max', axis=1, inplace=True)
    return df

def scale_data(df, scaler=None):
    df['cycle_norm'] = df['cycle']
    # Use only sensors with variance (Removing s1, s10, s18, s19)
    cols_normalize = ['setting1', 'setting2', 'setting3', 'cycle_norm'] + \
                     ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    
    if scaler is None:
        scaler = preprocessing.MinMaxScaler()
        df[cols_normalize] = scaler.fit_transform(df[cols_normalize])
    else:
        df[cols_normalize] = scaler.transform(df[cols_normalize])
        
    return df, scaler

def gen_sequence(id_df, seq_length, seq_cols):
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]

def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]