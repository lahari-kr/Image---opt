# ==============================
# Transformer vs Baseline Comparison
# ==============================
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Flatten, Reshape
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Synthetic dataset (you can replace this with your dataset)
# ---------------------------
np.random.seed(42)
X = np.random.rand(1000, 20)
y = np.random.randint(0, 3, 1000)  # 3 classes

y = tf.keras.utils.to_categorical(y, 3)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# Baseline Model (Dense NN)
# ---------------------------
def create_baseline_model(input_dim, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    return model

# ---------------------------
# Transformer Model
# ---------------------------
def create_transformer_model(input_dim, num_classes, embed_dim=64, num_heads=4, ff_dim=128):
    inputs = Input(shape=(input_dim,))
    x = Dense(embed_dim)(inputs)
    x = Reshape((1, embed_dim))(x)

    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attn_output)

    ffn_output = Sequential([
        Dense(ff_dim, activation='relu'),
        Dropout(0.3),
        Dense(embed_dim)
    ])(x)
    x = LayerNormalization(epsilon=1e-6)(x + ffn_output)

    x = Flatten()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

# ---------------------------
# Training and Evaluation
# ---------------------------
learning_rates = [0.0001, 0.01, 0.5]
results = []

# Create folder for plots
os.makedirs("plots", exist_ok=True)
sns.set(style="whitegrid", font_scale=1.2)

for lr in learning_rates:
    print(f"\nTraining models with Learning Rate = {lr}")

    # Baseline model
    baseline = create_baseline_model(X_train.shape[1], y_train.shape[1])
    baseline.compile(optimizer=Adam(learning_rate=lr),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    hist_base = baseline.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=10, batch_size=32, verbose=0
    )

    # Transformer model
    transformer = create_transformer_model(X_train.shape[1], y_train.shape[1])
    transformer.compile(optimizer=Adam(learning_rate=lr),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    hist_trans = transformer.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=10, batch_size=32, verbose=0
    )

    # Final evaluation
    base_train_loss, base_train_acc = baseline.evaluate(X_train, y_train, verbose=0)
    base_val_loss, base_val_acc = baseline.evaluate(X_val, y_val, verbose=0)

    trans_train_loss, trans_train_acc = transformer.evaluate(X_train, y_train, verbose=0)
    trans_val_loss, trans_val_acc = transformer.evaluate(X_val, y_val, verbose=0)

    results.append({
        "Model": "Baseline", "LR": lr,
        "Train Acc": base_train_acc, "Val Acc": base_val_acc,
        "Train Loss": base_train_loss, "Val Loss": base_val_loss
    })
    results.append({
        "Model": "Transformer", "LR": lr,
        "Train Acc": trans_train_acc, "Val Acc": trans_val_acc,
        "Train Loss": trans_train_loss, "Val Loss": trans_val_loss
    })

    # Plot accuracy
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist_base.history['accuracy'], 'o-', label='Baseline')
    plt.plot(hist_trans.history['accuracy'], 's-', label='Transformer')
    plt.title(f"Training Accuracy (LR={lr})")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist_base.history['val_accuracy'], 'o--', label='Baseline')
    plt.plot(hist_trans.history['val_accuracy'], 's--', label='Transformer')
    plt.title(f"Validation Accuracy (LR={lr})")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/accuracy_lr_{lr}.png", dpi=300)
    plt.close()

    # Plot loss
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist_base.history['loss'], 'o-', label='Baseline')
    plt.plot(hist_trans.history['loss'], 's-', label='Transformer')
    plt.title(f"Training Loss (LR={lr})")
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist_base.history['val_loss'], 'o--', label='Baseline')
    plt.plot(hist_trans.history['val_loss'], 's--', label='Transformer')
    plt.title(f"Validation Loss (LR={lr})")
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/loss_lr_{lr}.png", dpi=300)
    plt.close()

# ---------------------------
# Display Results
# ---------------------------
results_df = pd.DataFrame(results)
print("\nFinal Comparison Table:")
print(results_df.to_string(index=False))
