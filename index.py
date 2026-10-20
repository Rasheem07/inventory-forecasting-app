
import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. CONFIGURATION
# ==========================================

SEED = 42
LOOKBACK = 30
EPOCHS = 30
BATCH_SIZE = 64

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==========================================
# 2. LOAD DATASETS
# ==========================================

train = pd.read_csv("train.csv", parse_dates=["date"])
test = pd.read_csv("test.csv", parse_dates=["date"])

train = train.sort_values(["store", "item", "date"]).reset_index(drop=True)
test = test.sort_values(["store", "item", "date"]).reset_index(drop=True)

print("Training dataset:", train.shape)
print("Testing dataset:", test.shape)

print("\nTraining data:")
print(train.head())

# ==========================================
# 3. CHRONOLOGICAL SPLIT
# ==========================================
# Keep the final 90 days of training history
# for validation, avoiding random splitting
# across future and past observations.

validation_start = train["date"].max() - pd.Timedelta(days=89)

train_part = train[train["date"] < validation_start].copy()
val_part = train[train["date"] >= validation_start].copy()

# Fit the scaler ONLY using training data.
scaler = MinMaxScaler()
scaler.fit(train_part[["sales"]])

train["sales_scaled"] = scaler.transform(train[["sales"]])

# ==========================================
# 4. CREATE RNN SEQUENCES
# ==========================================

def create_sequences(data, lookback, validation_start):
    X_train, y_train = [], []
    X_val, y_val = [], []
    val_info = []

    for (store, item), group in data.groupby(["store", "item"]):
        group = group.sort_values("date")

        values = group["sales_scaled"].to_numpy(dtype=np.float32)
        dates = group["date"].to_numpy()
        actual_sales = group["sales"].to_numpy()

        for i in range(lookback, len(group)):
            sequence = values[i - lookback:i].reshape(lookback, 1)
            target = values[i]

            if dates[i] < np.datetime64(validation_start):
                X_train.append(sequence)
                y_train.append(target)
            else:
                X_val.append(sequence)
                y_val.append(target)
                val_info.append({
                    "date": dates[i],
                    "store": store,
                    "item": item,
                    "actual": actual_sales[i]
                })

    return (
        np.array(X_train, dtype=np.float32),
        np.array(y_train, dtype=np.float32),
        np.array(X_val, dtype=np.float32),
        np.array(y_val, dtype=np.float32),
        pd.DataFrame(val_info)
    )


X_train, y_train, X_val, y_val, val_info = create_sequences(
    train, LOOKBACK, validation_start
)

print("\nX_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

# ==========================================
# 5. BUILD SIMPLE RNN MODEL
# ==========================================

model = Sequential([
    Input(shape=(LOOKBACK, 1)),

    SimpleRNN(
        units=64,
        activation="tanh",
        return_sequences=True
    ),

    Dropout(0.2),

    SimpleRNN(
        units=32,
        activation="tanh"
    ),

    Dense(16, activation="relu"),

    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

model.summary()

# ==========================================
# 6. TRAIN MODEL
# ==========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    shuffle=True,
    verbose=1
)

# ==========================================
# 7. EVALUATE MODEL
# ==========================================

pred_scaled = model.predict(X_val, batch_size=512)

predictions = scaler.inverse_transform(pred_scaled).flatten()
actual = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

# Demand cannot be negative.
predictions = np.maximum(predictions, 0)

mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))

print("\nMODEL PERFORMANCE")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")

val_info["predicted"] = predictions

# ==========================================
# 8. PLOT TRAINING HISTORY
# ==========================================

plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("RNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# 9. ACTUAL VS PREDICTED SALES
# ==========================================

# Display a single store-item series,
# not a mix of different products.

sample = val_info[
    (val_info["store"] == 1) &
    (val_info["item"] == 1)
].sort_values("date")

plt.figure(figsize=(12, 5))

plt.plot(
    sample["date"],
    sample["actual"],
    label="Actual Sales"
)

plt.plot(
    sample["date"],
    sample["predicted"],
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Sales - Store 1, Item 1")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# 10. FORECAST KAGGLE TEST DATA
# ==========================================
# The test file contains future dates without
# sales values. Predict them recursively.
#
# Each store-item maintains its own historical
# sequence. Earlier predictions feed into
# later predictions.

# Group test rows by store and item.
test_predictions = np.zeros(len(test), dtype=np.float32)

for (store, item), group in test.groupby(["store", "item"], sort=False):

    past = train[
        (train["store"] == store) &
        (train["item"] == item)
    ].sort_values("date")

    window = past["sales_scaled"].to_numpy(
        dtype=np.float32
    )[-LOOKBACK:].copy()

    if len(window) < LOOKBACK:
        raise ValueError(
            f"Not enough history for store={store}, item={item}"
        )

    # Predict dates in chronological order.
    for idx in group.sort_values("date").index:

        X = window.reshape(1, LOOKBACK, 1)

        next_scaled = model(X, training=False).numpy()[0, 0]

        # Convert forecast back to actual units.
        next_sales = scaler.inverse_transform(
            np.array([[next_scaled]])
        )[0, 0]

        next_sales = max(0, float(next_sales))

        test_predictions[idx] = next_sales

        # Feed the scaled forecast back into
        # the next 30-day input window.
        next_scaled = scaler.transform(
            pd.DataFrame({"sales": [next_sales]})
        )[0, 0]

        window = np.concatenate([
            window[1:],
            np.array([next_scaled], dtype=np.float32)
        ])

# ==========================================
# 11. SAVE OUTPUT
# ==========================================

submission = pd.DataFrame({
    "id": test["id"],
    "sales": test_predictions
})

submission.to_csv("submission.csv", index=False)
model.save("inventory_rnn.keras")

val_info.to_csv("validation_predictions.csv", index=False)

print("\nSaved files:")
print("submission.csv")
print("inventory_rnn.keras")
print("validation_predictions.csv")

print("\nSample forecasts:")
print(submission.head())
