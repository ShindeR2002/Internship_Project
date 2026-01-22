import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

def build_lstm_model(seq_array, label_array):
    features_count = seq_array.shape[2]
    nb_out = label_array.shape[1]

    model = Sequential()
    # Match notebook's simple but deep stack
    model.add(LSTM(input_shape=(50, features_count), units=100, return_sequences=True))
    model.add(Dropout(0.4)) 

    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.4))

    model.add(Dense(units=nb_out, activation='sigmoid'))

    # Use a slightly lower learning rate for more stable convergence
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model