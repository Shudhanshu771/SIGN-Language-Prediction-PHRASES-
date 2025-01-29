from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import os
import numpy as np

# Define the label map for actions
label_map = {label: num for num, label in enumerate(actions)}
# print(label_map)

# Initialize sequences and labels
sequences, labels = [], []

# Load data for each action and sequence
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            # Load the keypoints data, allowing for pickling
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
            
            # Check shape consistency
            if res.shape != (63,):  # Expected shape: 21 keypoints * 3 coordinates (x, y, z)
                print(f"Unexpected shape: {res.shape} at Action: {action}, Sequence: {sequence}, Frame: {frame_num}")
                continue  # Skip frames with unexpected shapes
            
            window.append(res)
        
        # Ensure window length is consistent with sequence_length
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])
        else:
            # Pad sequences if they are too short
            while len(window) < sequence_length:
                window.append(np.zeros(63))  # Pad with zeros, adjust based on expected keypoint shape
            # Or trim sequences if they are too long
            window = window[:sequence_length]
            
            # Append after adjustment
            sequences.append(window)
            labels.append(label_map[action])

# Convert to NumPy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Set up TensorBoard for monitoring
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# Print model summary
model.summary()

# Save the model architecture and weights
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')
