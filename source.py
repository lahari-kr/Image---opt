import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, \
    MultiHeadAttention, Flatten, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

# Load & preprocess
df = pd.read_csv("synthetic_book_dataset.csv")
titleencoder = LabelEncoder()
genreencoder = LabelEncoder()
descriptionencoder = OneHotEncoder(sparse_output=False)
df['TitleEncoded'] = titleencoder.fit_transform(df['Title'])
df['GenreEncoded'] = genreencoder.fit_transform(df['Genre'])
desc_encoded = descriptionencoder.fit_transform(df['Description'].values.reshape(-1, 1))
desc_encoded_df = pd.DataFrame(desc_encoded, columns=descriptionencoder.categories_[0])
df = pd.concat([df, desc_encoded_df], axis=1)
X = np.array(df[['TitleEncoded', 'GenreEncoded']])
y = np.array(df[desc_encoded_df.columns])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline
def build_baseline_model(input_shape, output_dim):
    return Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(output_dim, activation='softmax')
    ])

# Transformer
def build_transformer_model(input_shape, output_dim,
                            d_model=256, num_heads=8, ff_dim=512, dropout=0.3):
    inputs = Input(shape=(input_shape,))
    x = Reshape((1, input_shape))(inputs)
    x = Dense(d_model, activation='relu')(x)
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = Flatten()(attn)
    x = LayerNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = LayerNormalization()(x)
    outputs = Dense(output_dim, activation='softmax')(x)
    return Model(inputs, outputs)

def save_learning_plots(history, lr, folder='plots'):
    print(f"Creating accuracy graph for learning rate {lr}")
    print(f"Creating loss graph for learning rate {lr}")

    plt.figure()
    plt.plot(history.history['accuracy'], 'o-', label='Train')
    plt.plot(history.history['val_accuracy'], 's--', label='Validation')
    plt.title(f'Accuracy for Learning Rate {lr}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{folder}/lr_{lr}_accuracy.jpg')
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'], 'o-', label='Train')
    plt.plot(history.history['val_loss'], 's--', label='Validation')
    plt.title(f'Loss for Learning Rate {lr}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{folder}/lr_{lr}_loss.jpg')
    plt.close()

def save_combined_plots(baseline_hist, transformer_hist, best_lr, folder='plots'):
    print("\nGenerating and saving combined comparison graphs...")
    plt.figure(figsize=(10,5))
    plt.plot(baseline_hist.history['accuracy'], 'o-', label='Baseline Train')
    plt.plot(baseline_hist.history['val_accuracy'], 's--', label='Baseline Val')
    plt.plot(transformer_hist.history['accuracy'], 'o-', label=f'Transformer Train (LR={best_lr})')
    plt.plot(transformer_hist.history['val_accuracy'], 's--', label=f'Transformer Val (LR={best_lr})')
    plt.title('Combined Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{folder}/combined_accuracy_comparison.jpg')
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(baseline_hist.history['loss'], 'o-', label='Baseline Train')
    plt.plot(baseline_hist.history['val_loss'], 's--', label='Baseline Val')
    plt.plot(transformer_hist.history['loss'], 'o-', label=f'Transformer Train (LR={best_lr})')
    plt.plot(transformer_hist.history['val_loss'], 's--', label=f'Transformer Val (LR={best_lr})')
    plt.title('Combined Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{folder}/combined_loss_comparison.jpg')
    plt.close()

print("Training baseline model (learning rate 0.001)...")
baseline_model = build_baseline_model(X_train.shape[1], y_train.shape[1])
baseline_model.compile(optimizer=Adam(0.001),
                       loss='categorical_crossentropy', metrics=['accuracy'])
baseline_history = baseline_model.fit(X_train, y_train,
                                     validation_data=(X_test, y_test),
                                     epochs=10, batch_size=32, verbose=1)

print("\nTraining Transformer model with different learning rates and adaptive scheduler...")
learning_rates = [0.0001, 0.01, 0.5]
transformer_histories = {}

for lr in learning_rates:
    print(f"\nTraining model with learning rate: {lr}")
    scheduler = CosineDecayRestarts(lr, first_decay_steps=5, t_mul=2.0, m_mul=0.9, alpha=1e-5)
    transformer_model = build_transformer_model(X_train.shape[1], y_train.shape[1])
    transformer_model.compile(optimizer=Adam(learning_rate=scheduler),
                              loss='categorical_crossentropy', metrics=['accuracy'])
    history = transformer_model.fit(X_train, y_train,
                                    validation_data=(X_test, y_test),
                                    epochs=10, batch_size=32, verbose=1)
    transformer_histories[lr] = history

    print(f"Learning Rate: {lr}")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")

    save_learning_plots(history, lr)

best_lr = max(transformer_histories.items(), key=lambda x: x[1].history['val_accuracy'][-1])[0]

print("\n===== BEST LEARNING RATE ANALYSIS =====")
print(f"Best learning rate by validation accuracy: {best_lr} "
      f"(accuracy: {transformer_histories[best_lr].history['val_accuracy'][-1]:.4f})")

save_combined_plots(baseline_history, transformer_histories[best_lr], best_lr)

print("\nSample Input:")
sample_title = 'Journey to Mars'
sample_genre = 'Science Fiction'
print(f"Title: {sample_title}, Genre: {sample_genre}")

sample_input = np.array([[titleencoder.transform([sample_title])[0],
                          genreencoder.transform([sample_genre])[0]]])

for lr, history in transformer_histories.items():
    model = build_transformer_model(X_train.shape[1], y_train.shape[1])
    scheduler = CosineDecayRestarts(lr, first_decay_steps=5, t_mul=2.0, m_mul=0.9, alpha=1e-5)
    model.compile(optimizer=Adam(learning_rate=scheduler), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    prediction = model.predict(sample_input)
    generated_desc = descriptionencoder.inverse_transform(prediction)[0][0]
    print(f"\nLearning Rate: {lr}")
    print(f"Generated Description: [\"{generated_desc}\"]")
    if lr == best_lr:
        print("*** BEST MODEL BASED ON VALIDATION ACCURACY ***")

# Figure1: Compare all LR accuracies/losses for both models + show plots one-by-one
plt.style.use('ggplot')
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

# Accuracy plot (all LRs)
for lr, hist in transformer_histories.items():
    axs[0].plot(hist.history['accuracy'], label=f'Transformer LR={lr}')
axs[0].plot(baseline_history.history['accuracy'], label='Baseline')
axs[0].set_title('Training Accuracy Comparison (All LRs)')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Accuracy')
axs[0].legend()
axs[0].grid(True)

# Loss plot (all LRs)
for lr, hist in transformer_histories.items():
    axs[1].plot(hist.history['loss'], label=f'Transformer LR={lr}')
axs[1].plot(baseline_history.history['loss'], label='Baseline')
axs[1].set_title('Training Loss Comparison (All LRs)')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig('plots/figure1_all_LR_comparison.jpg')
plt.show()

# Show individual plot images (simulate source.py style)
import matplotlib.image as mpimg

plot_files = [
    'plots/lr_0.0001_accuracy.jpg', 'plots/lr_0.0001_loss.jpg',
    'plots/lr_0.01_accuracy.jpg', 'plots/lr_0.01_loss.jpg',
    'plots/lr_0.5_accuracy.jpg', 'plots/lr_0.5_loss.jpg',
    'plots/combined_accuracy_comparison.jpg', 'plots/combined_loss_comparison.jpg',
]

for file in plot_files:
    img = mpimg.imread(file)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Showing: {file.split('/')[-1]}")
    plt.show()


# Summary comparison table for all learning rates and baseline
print("\nComparison Table (Final Epoch):")
print(f"{'Model':<20} {'LR':<10} {'Train Acc':<12} {'Val Acc':<12} {'Train Loss':<12} {'Val Loss':<12}")
print("-"*70)
print(f"{'Baseline':<20} {'0.001':<10} "
      f"{baseline_history.history['accuracy'][-1]:<12.4f} {baseline_history.history['val_accuracy'][-1]:<12.4f} "
      f"{baseline_history.history['loss'][-1]:<12.4f} {baseline_history.history['val_loss'][-1]:<12.4f}")

for lr, hist in transformer_histories.items():
    print(f"{'Transformer':<20} {lr:<10} "
          f"{hist.history['accuracy'][-1]:<12.4f} {hist.history['val_accuracy'][-1]:<12.4f} "
          f"{hist.history['loss'][-1]:<12.4f} {hist.history['val_loss'][-1]:<12.4f}")
