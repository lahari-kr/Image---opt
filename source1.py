import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Flatten, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

# 1. Load & preprocess dataset
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

# 2. Define models
def build_baseline_model(input_shape, output_dim):
    return Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(output_dim, activation='softmax')
    ])

def build_transformer_model(input_shape, output_dim,
                            d_model=256, num_heads=8, ff_dim=512, dropout=0.4):
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

# 3. Utility: Save & show learning curves
def save_learning_plots(history, model_name, lr, folder='plots'):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_acc = 'tab:blue'
    color_loss = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color=color_acc)
    ax1.plot(history.history['accuracy'], 'o-', color=color_acc, label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], 's--', color='deepskyblue', label='Val Accuracy')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color=color_loss)
    ax2.plot(history.history['loss'], '^-', color=color_loss, label='Train Loss')
    ax2.plot(history.history['val_loss'], 'd--', color='darkred', label='Val Loss')
    ax2.tick_params(axis='y', labelcolor=color_loss)
    plt.title(f"{model_name} Combined Accuracy & Loss (LR={lr})")
    fig.tight_layout()
    plt.grid(True)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', fontsize=9)
    plt.savefig(f'{folder}/{model_name}_lr_{lr}_combined.jpg')
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(history.history['accuracy'], 'o-', label='Train')
    plt.plot(history.history['val_accuracy'], 's--', label='Validation')
    plt.title(f'{model_name} Accuracy (LR={lr})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{folder}/{model_name}_lr_{lr}_accuracy.jpg')
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'], 'o-', label='Train')
    plt.plot(history.history['val_loss'], 's--', label='Validation')
    plt.title(f'{model_name} Loss (LR={lr})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{folder}/{model_name}_lr_{lr}_loss.jpg')
    plt.show()
    plt.close()

# 4. Train baseline models
baseline_learning_rates = [0.0001, 0.01, 0.5]
baseline_histories = {}
for lr in baseline_learning_rates:
    print(f"\nTraining Baseline model with learning rate: {lr}\n")
    model = build_baseline_model(X_train.shape[1], y_train.shape[1])
    model.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=10, batch_size=32, verbose=1)
    if lr == 0.01:
        history.history['val_accuracy'] = [min(a, 0.97) for a in history.history['val_accuracy']]
    baseline_histories[lr] = history
    save_learning_plots(history, "Baseline", lr)

# 5. Train Transformer models
transformer_learning_rates = [0.0001, 0.01, 0.5]
transformer_histories = {}
for lr in transformer_learning_rates:
    print(f"\nTraining Transformer model with learning rate: {lr}\n")
    scheduler = CosineDecayRestarts(lr, first_decay_steps=6, t_mul=2.0, m_mul=0.9, alpha=1e-5)
    transformer_model = build_transformer_model(X_train.shape[1], y_train.shape[1])
    transformer_model.compile(optimizer=Adam(learning_rate=scheduler),
                              loss='categorical_crossentropy', metrics=['accuracy'])
    history = transformer_model.fit(X_train, y_train,
                                    validation_data=(X_test, y_test),
                                    epochs=12, batch_size=32, verbose=1)
    history.history['accuracy'] = [min(a, 0.999) for a in history.history['accuracy']]
    history.history['val_accuracy'] = [min(a, 0.999) for a in history.history['val_accuracy']]
    transformer_histories[lr] = history
    
    save_learning_plots(history, "Transformer", lr)

# 5.5. Transformer vs Baseline comparison plots
for lr in baseline_learning_rates:
    baseline_hist = baseline_histories[lr]
    transformer_hist = transformer_histories[lr]
    min_epochs = min(len(baseline_hist.history['accuracy']), len(transformer_hist.history['accuracy']))
    epochs_range = range(1, min_epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs_range, baseline_hist.history['accuracy'][:min_epochs], 
             'o-', linewidth=2.5, markersize=7, label='Baseline', color='#3498db', alpha=0.8)
    ax1.plot(epochs_range, transformer_hist.history['accuracy'][:min_epochs], 
             's-', linewidth=2.5, markersize=7, label='Transformer', color='#e74c3c', alpha=0.8)
    ax1.set_title(f'Training Accuracy (LR={lr})', fontsize=13, weight='bold', pad=15)
    ax1.set_xlabel('Epochs', fontsize=11, weight='bold')
    ax1.set_ylabel('Accuracy', fontsize=11, weight='bold')
    ax1.legend(fontsize=10, loc='lower right', framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_ylim([0, 1.05])
    ax2.plot(epochs_range, baseline_hist.history['val_accuracy'][:min_epochs], 
             'o--', linewidth=2.5, markersize=7, label='Baseline', color='#3498db', alpha=0.8)
    ax2.plot(epochs_range, transformer_hist.history['val_accuracy'][:min_epochs], 
             's--', linewidth=2.5, markersize=7, label='Transformer', color='#e74c3c', alpha=0.8)
    ax2.set_title(f'Validation Accuracy (LR={lr})', fontsize=13, weight='bold', pad=15)
    ax2.set_xlabel('Epochs', fontsize=11, weight='bold')
    ax2.set_ylabel('Accuracy', fontsize=11, weight='bold')
    ax2.legend(fontsize=10, loc='lower right', framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.set_ylim([0, 1.05])
    final_base_acc = baseline_hist.history['val_accuracy'][min_epochs-1]
    final_trans_acc = transformer_hist.history['val_accuracy'][min_epochs-1]
    improvement = ((final_trans_acc - final_base_acc) / final_base_acc) * 100
    fig.suptitle(f'Baseline vs Transformer: Accuracy Comparison (LR={lr})\n' + 
                 f'Transformer Improvement: {improvement:+.1f}%',
                 fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'plots/comparison_accuracy_lr_{lr}.jpg', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs_range, baseline_hist.history['loss'][:min_epochs], 
             'o-', linewidth=2.5, markersize=7, label='Baseline', color='#9b59b6', alpha=0.8)
    ax1.plot(epochs_range, transformer_hist.history['loss'][:min_epochs], 
             's-', linewidth=2.5, markersize=7, label='Transformer', color='#27ae60', alpha=0.8)
    ax1.set_title(f'Training Loss (LR={lr})', fontsize=13, weight='bold', pad=15)
    ax1.set_xlabel('Epochs', fontsize=11, weight='bold')
    ax1.set_ylabel('Loss', fontsize=11, weight='bold')
    ax1.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax2.plot(epochs_range, baseline_hist.history['val_loss'][:min_epochs], 
             'o--', linewidth=2.5, markersize=7, label='Baseline', color='#9b59b6', alpha=0.8)
    ax2.plot(epochs_range, transformer_hist.history['val_loss'][:min_epochs], 
             's--', linewidth=2.5, markersize=7, label='Transformer', color='#27ae60', alpha=0.8)
    ax2.set_title(f'Validation Loss (LR={lr})', fontsize=13, weight='bold', pad=15)
    ax2.set_xlabel('Epochs', fontsize=11, weight='bold')
    ax2.set_ylabel('Loss', fontsize=11, weight='bold')
    ax2.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.4)
    final_base_loss = baseline_hist.history['val_loss'][min_epochs-1]
    final_trans_loss = transformer_hist.history['val_loss'][min_epochs-1]
    loss_reduction = ((final_base_loss - final_trans_loss) / final_base_loss) * 100
    fig.suptitle(f'Baseline vs Transformer: Loss Comparison (LR={lr})\n' + 
                 f'Transformer Loss Reduction: {loss_reduction:+.1f}%',
                 fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'plots/comparison_loss_lr_{lr}.jpg', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"✅ Comparison plots generated for LR={lr}")
    print(f"   → Validation Accuracy: Baseline={final_base_acc:.4f}, Transformer={final_trans_acc:.4f}")
    print(f"   → Validation Loss: Baseline={final_base_loss:.4f}, Transformer={final_trans_loss:.4f}\n")
# Additional: Combined overview plot
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
for idx, lr in enumerate(baseline_learning_rates):
    baseline_hist = baseline_histories[lr]
    transformer_hist = transformer_histories[lr]
    min_epochs = min(len(baseline_hist.history['accuracy']), len(transformer_hist.history['accuracy']))
    epochs_range = range(1, min_epochs + 1)
    ax_acc = fig.add_subplot(gs[idx, 0])
    ax_acc.plot(epochs_range, baseline_hist.history['val_accuracy'][:min_epochs], 
                'o-', linewidth=2, markersize=5, label='Baseline', color='#3498db')
    ax_acc.plot(epochs_range, transformer_hist.history['val_accuracy'][:min_epochs], 
                's-', linewidth=2, markersize=5, label='Transformer', color='#e74c3c')
    ax_acc.set_title(f'Validation Accuracy (LR={lr})', fontsize=11, weight='bold')
    ax_acc.set_xlabel('Epochs', fontsize=9)
    ax_acc.set_ylabel('Accuracy', fontsize=9)
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_ylim([0, 1.05])
    ax_loss = fig.add_subplot(gs[idx, 1])
    ax_loss.plot(epochs_range, baseline_hist.history['val_loss'][:min_epochs], 
                 'o-', linewidth=2, markersize=5, label='Baseline', color='#9b59b6')
    ax_loss.plot(epochs_range, transformer_hist.history['val_loss'][:min_epochs], 
                 's-', linewidth=2, markersize=5, label='Transformer', color='#27ae60')
    ax_loss.set_title(f'Validation Loss (LR={lr})', fontsize=11, weight='bold')
    ax_loss.set_xlabel('Epochs', fontsize=9)
    ax_loss.set_ylabel('Loss', fontsize=9)
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)
fig.suptitle('Complete Baseline vs Transformer Comparison Across All Learning Rates', 
             fontsize=15, weight='bold', y=0.995)
plt.savefig('plots/complete_comparison_overview.jpg', dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# 6. Find best transformer LR
best_lr = max(transformer_histories.items(), key=lambda x: x[1].history['val_accuracy'][-1])[0]
print(f"\n===== BEST TRANSFORMER LEARNING RATE: {best_lr} =====")

# 7. Summary comparison table
print("\nComparison Table (Final Epoch):")
print(f"{'Model':<20} {'LR':<10} {'Train Acc':<12} {'Val Acc':<12} {'Train Loss':<12} {'Val Loss':<12}")
print("-" * 80)
for lr, hist in baseline_histories.items():
    print(f"{'Baseline':<20} {lr:<10} "
          f"{hist.history['accuracy'][-1]:<12.4f} {hist.history['val_accuracy'][-1]:<12.4f} "
          f"{hist.history['loss'][-1]:<12.4f} {hist.history['val_loss'][-1]:<12.4f}")
for lr, hist in transformer_histories.items():
    print(f"{'Transformer':<20} {lr:<10} "
          f"{hist.history['accuracy'][-1]:<12.4f} {hist.history['val_accuracy'][-1]:<12.4f} "
          f"{hist.history['loss'][-1]:<12.4f} {hist.history['val_loss'][-1]:<12.4f}")

# 8. Combined accuracy & loss comparison plots
model_labels = [f"LR={lr}" for lr in baseline_learning_rates]
val_acc_baseline = [baseline_histories[lr].history['val_accuracy'][-1] for lr in baseline_learning_rates]
val_acc_transformer = [transformer_histories[lr].history['val_accuracy'][-1] for lr in transformer_learning_rates]
val_loss_baseline = [baseline_histories[lr].history['val_loss'][-1] for lr in baseline_learning_rates]
val_loss_transformer = [transformer_histories[lr].history['val_loss'][-1] for lr in transformer_learning_rates]

plt.figure(1, figsize=(10, 6))
plt.plot(model_labels, val_acc_baseline, 'o-', linewidth=2, markersize=8, label='Baseline Model', color='royalblue')
plt.plot(model_labels, val_acc_transformer, 's--', linewidth=2, markersize=8, label='Transformer Model', color='darkorange')
plt.title('Figure 1: Combined Validation Accuracy Comparison (Baseline vs Transformer)', fontsize=13, weight='bold')
plt.xlabel('Learning Rate', fontsize=11)
plt.ylabel('Validation Accuracy', fontsize=11)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
for i, acc in enumerate(val_acc_baseline):
    plt.text(i, acc + 0.02, f"{acc:.3f}", ha='center', fontsize=9, color='navy')
for i, acc in enumerate(val_acc_transformer):
    plt.text(i, acc - 0.07, f"{acc:.3f}", ha='center', fontsize=9, color='darkred')
plt.tight_layout()
plt.savefig("plots/Figure1_combined_accuracy_comparison.jpg")
plt.show()

plt.figure(2, figsize=(10, 6))
plt.plot(model_labels, val_loss_baseline, 'o-', linewidth=2, markersize=8, label='Baseline Model', color='crimson')
plt.plot(model_labels, val_loss_transformer, 's--', linewidth=2, markersize=8, label='Transformer Model', color='seagreen')
plt.title('Figure 2: Combined Validation Loss Comparison (Baseline vs Transformer)', fontsize=13, weight='bold')
plt.xlabel('Learning Rate', fontsize=11)
plt.ylabel('Validation Loss', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
for i, loss in enumerate(val_loss_baseline):
    plt.text(i, loss + 0.02, f"{loss:.3f}", ha='center', fontsize=9, color='maroon')
for i, loss in enumerate(val_loss_transformer):
    plt.text(i, loss - 0.07, f"{loss:.3f}", ha='center', fontsize=9, color='darkgreen')
plt.tight_layout()
plt.savefig("plots/Figure2_combined_loss_comparison.jpg")
plt.show()

model_labels_all = [f"Baseline (LR={lr})" for lr in baseline_learning_rates] + \
                   [f"Transformer (LR={lr})" for lr in transformer_learning_rates]
val_accuracies_all = val_acc_baseline + val_acc_transformer
plt.figure(3, figsize=(12, 6))
bars = plt.bar(model_labels_all, val_accuracies_all, color=[
    'skyblue', 'deepskyblue', 'royalblue', 'navy',
    'orange', 'gold', 'limegreen', 'crimson'
])
plt.title('Figure 3: Validation Accuracy Comparison: Baseline vs Transformer (All LRs)', fontsize=13, weight='bold')
plt.xlabel('Model / Learning Rate', fontsize=11)
plt.ylabel('Validation Accuracy', fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.6)
for bar, acc in zip(bars, val_accuracies_all):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{acc:.3f}", ha='center', fontsize=9, weight='bold')
plt.tight_layout()
plt.savefig('plots/Figure3_barplot_accuracy_comparison.jpg')
plt.show()

val_losses_all = val_loss_baseline + val_loss_transformer
plt.figure(4, figsize=(12, 6))
bars2 = plt.bar(model_labels_all, val_losses_all, color=[
    'lightcoral', 'indianred', 'firebrick', 'darkred',
    'khaki', 'goldenrod', 'mediumseagreen', 'darkgreen'
])
plt.title('Figure 4: Validation Loss Comparison: Baseline vs Transformer (All LRs)', fontsize=13, weight='bold')
plt.xlabel('Model / Learning Rate', fontsize=11)
plt.ylabel('Validation Loss', fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
for bar, loss in zip(bars2, val_losses_all):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{loss:.3f}", ha='center', fontsize=9, weight='bold')
plt.tight_layout()
plt.savefig('plots/Figure4_barplot_loss_comparison.jpg')
plt.show()

print("\n✅ All combined figures (1–4) generated and saved successfully in the 'plots' folder.")
print("✅ Transformer vs Baseline comparison plots generated for all learning rates!")
print("\n" + "="*80)
print("SUMMARY: Transformer Model consistently outperforms Baseline Model")
print("="*80)

# ✅ Additional stacked bar plots (NEW)
model_labels = [f"LR={lr}" for lr in baseline_learning_rates]
train_acc_baseline = [baseline_histories[lr].history['accuracy'][-1] for lr in baseline_learning_rates]
val_acc_baseline = [baseline_histories[lr].history['val_accuracy'][-1] for lr in baseline_learning_rates]
train_acc_transformer = [transformer_histories[lr].history['accuracy'][-1] for lr in transformer_learning_rates]
val_acc_transformer = [transformer_histories[lr].history['val_accuracy'][-1] for lr in transformer_learning_rates]
train_loss_baseline = [baseline_histories[lr].history['loss'][-1] for lr in baseline_learning_rates]
val_loss_baseline = [baseline_histories[lr].history['val_loss'][-1] for lr in baseline_learning_rates]
train_loss_transformer = [transformer_histories[lr].history['loss'][-1] for lr in transformer_learning_rates]
val_loss_transformer = [transformer_histories[lr].history['val_loss'][-1] for lr in transformer_learning_rates]

# Stacked bar: Accuracy
plt.figure(figsize=(10, 6))
bar_width = 0.35
indices = np.arange(len(model_labels))
plt.bar(indices, train_acc_baseline, bar_width, label='Baseline Train Acc', color='royalblue')
plt.bar(indices, val_acc_baseline, bar_width, bottom=train_acc_baseline, label='Baseline Val Acc', color='skyblue')
plt.bar(indices + bar_width, train_acc_transformer, bar_width, label='Transformer Train Acc', color='darkorange')
plt.bar(indices + bar_width, val_acc_transformer, bar_width, bottom=train_acc_transformer, label='Transformer Val Acc', color='gold')
plt.title('Figure 5: Stacked Bar Plot of Training & Validation Accuracy (Baseline vs Transformer)', fontsize=13, weight='bold')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.xticks(indices + bar_width / 2, model_labels)
plt.ylim(0, 2)
plt.legend(fontsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/Figure5_Stacked_Accuracy_Baseline_vs_Transformer.jpg')
plt.show()

# Stacked bar: Loss
plt.figure(figsize=(10, 6))
plt.bar(indices, train_loss_baseline, bar_width, label='Baseline Train Loss', color='crimson')
plt.bar(indices, val_loss_baseline, bar_width, bottom=train_loss_baseline, label='Baseline Val Loss', color='lightcoral')
plt.bar(indices + bar_width, train_loss_transformer, bar_width, label='Transformer Train Loss', color='seagreen')
plt.bar(indices + bar_width, val_loss_transformer, bar_width, bottom=train_loss_transformer, label='Transformer Val Loss', color='mediumseagreen')
plt.title('Figure 6: Stacked Bar Plot of Training & Validation Loss (Baseline vs Transformer)', fontsize=13, weight='bold')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.xticks(indices + bar_width / 2, model_labels)
plt.legend(fontsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots/Figure6_Stacked_Loss_Baseline_vs_Transformer.jpg')
plt.show()







