# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load and preprocess dataset
# -----------------------------
max_features = 10000   # Use top 10,000 most common words
max_len = 200          # Truncate or pad each review to 200 words

print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(f"Training samples: {len(x_train)}")
print(f"Testing samples: {len(x_test)}")

# Pad sequences to ensure equal length
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

# -----------------------------
# 2. Build LSTM model
# -----------------------------
lstm_model = Sequential([
    Embedding(max_features, 128, input_length=max_len),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

print("\nTraining LSTM model...")
history_lstm = lstm_model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, y_test),
    verbose=1
)

# -----------------------------
# 3. Build BiLSTM model
# -----------------------------
bilstm_model = Sequential([
    Embedding(max_features, 128, input_length=max_len),
    Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)),
    Dense(1, activation='sigmoid')
])

bilstm_model.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

print("\nTraining BiLSTM model...")
history_bilstm = bilstm_model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, y_test),
    verbose=1
)

# -----------------------------
# 4. Evaluate and Compare
# -----------------------------
score_lstm = lstm_model.evaluate(x_test, y_test, verbose=0)
score_bilstm = bilstm_model.evaluate(x_test, y_test, verbose=0)

print("\n📊 Model Performance Comparison:")
print(f"LSTM Test Accuracy:   {score_lstm[1]*100:.2f}%")
print(f"BiLSTM Test Accuracy: {score_bilstm[1]*100:.2f}%")

# -----------------------------
# 5. Plot Accuracy and Loss
# -----------------------------
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history_lstm.history['accuracy'], label='LSTM Train')
plt.plot(history_lstm.history['val_accuracy'], label='LSTM Val')
plt.plot(history_bilstm.history['accuracy'], label='BiLSTM Train', linestyle='--')
plt.plot(history_bilstm.history['val_accuracy'], label='BiLSTM Val', linestyle='--')
plt.title('Model Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history_lstm.history['loss'], label='LSTM Train')
plt.plot(history_lstm.history['val_loss'], label='LSTM Val')
plt.plot(history_bilstm.history['loss'], label='BiLSTM Train', linestyle='--')
plt.plot(history_bilstm.history['val_loss'], label='BiLSTM Val', linestyle='--')
plt.title('Model Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
