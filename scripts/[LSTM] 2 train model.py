import os
import pickle
import numpy as np
import random
from glob import glob
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set seed for reproducibility
seed_value = 12345

def set_seed(seed_value=12345):
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

set_seed()

# Configuration
PKL_FOLDER = "./data/preprocessed"  # CHANGE THIS
WINDOW_SIZE = 48
STRIDE = 1
MIN_FALL_FRAMES = 6
MAX_FORWARD_FILL = 48
BATCH_SIZE = 32
EPOCHS = 50


def load_all_pkl_files(pkl_folder):
    """Load all preprocessed pkl files"""
    pkl_files = glob(os.path.join(pkl_folder, "*.pkl"))
    print(f"Found {len(pkl_files)} pkl files")
    
    all_data = []
    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            all_data.append(data)
    
    return all_data


def forward_fill_sequence(keypoints, valid_frames, max_fill=48):
    """
    Apply forward-fill with confidence=0 (max 48 frames)
    
    Args:
        keypoints: np.array(N, 17, 3) - [x_norm, y_norm, confidence]
        valid_frames: np.array(N) - boolean array
        max_fill: maximum frames to forward fill
    
    Returns:
        filled_keypoints: np.array(N, 17, 3)
    """
    filled = keypoints.copy()
    last_valid = None
    frames_since_valid = 0
    
    for i in range(len(keypoints)):
        if valid_frames[i]:
            # Valid frame
            last_valid = keypoints[i].copy()
            frames_since_valid = 0
        else:
            # Null frame
            frames_since_valid += 1
            
            if last_valid is not None and frames_since_valid <= max_fill:
                # Forward fill with confidence=0
                filled[i] = last_valid.copy()
                filled[i, :, 2] = 0.0  # Set all confidence to 0
            else:
                # Keep as [-1, -1, -1] (too long gap or no previous data)
                filled[i] = -1.0
    
    return filled


def label_window(frame_labels, min_fall_frames=6):
    """
    Label window based on consecutive fall frames
    
    Args:
        frame_labels: array of frame-level labels
        min_fall_frames: minimum consecutive fall frames
    
    Returns:
        'fall' or 'no_fall'
    """
    max_consecutive_falls = 0
    current_consecutive = 0
    
    for label in frame_labels:
        if label == 'fall':
            current_consecutive += 1
            max_consecutive_falls = max(max_consecutive_falls, current_consecutive)
        else:
            current_consecutive = 0
    
    if max_consecutive_falls >= min_fall_frames:
        return 'fall'
    else:
        return 'no_fall'


def create_windows(keypoints, frame_labels, window_size=48, stride=1, min_fall_frames=6):
    """
    Create sliding windows with stride
    
    Args:
        keypoints: np.array(N, 17, 3) - forward-filled keypoints
        frame_labels: np.array(N) - frame-level labels
        window_size: 48 frames
        stride: 1 frame
        min_fall_frames: minimum consecutive fall frames to label as "fall"
    
    Returns:
        windows: list of np.array(48, 17, 3)
        labels: list of 'fall' or 'no_fall'
    """
    windows = []
    labels = []
    
    total_frames = len(keypoints)
    
    for start_idx in range(0, total_frames - window_size + 1, stride):
        end_idx = start_idx + window_size
        
        # Extract window
        window = keypoints[start_idx:end_idx]
        window_frame_labels = frame_labels[start_idx:end_idx]
        
        # Label window
        window_label = label_window(window_frame_labels, min_fall_frames)
        
        windows.append(window)
        labels.append(window_label)
    
    return windows, labels


def balance_dataset(fall_windows, no_fall_windows, strategy='hybrid'):
    """
    Balance fall and no-fall samples
    
    Args:
        fall_windows: list of fall windows
        no_fall_windows: list of no-fall windows
        strategy: 'oversample', 'undersample', or 'hybrid'
    
    Returns:
        X: balanced windows
        y: balanced labels (0 or 1)
    """
    if strategy == 'oversample':
        # Oversample minority class (fall)
        fall_oversampled = resample(
            fall_windows,
            replace=True,
            n_samples=len(no_fall_windows),
            random_state=seed_value
        )
        X = np.array(fall_oversampled + no_fall_windows)
        y = np.array([1]*len(fall_oversampled) + [0]*len(no_fall_windows))
        
    elif strategy == 'undersample':
        # Undersample majority class (no_fall)
        no_fall_undersampled = resample(
            no_fall_windows,
            replace=False,
            n_samples=len(fall_windows),
            random_state=seed_value
        )
        X = np.array(fall_windows + no_fall_undersampled)
        y = np.array([1]*len(fall_windows) + [0]*len(no_fall_undersampled))
        
    elif strategy == 'hybrid':
        # Mix: oversample falls a bit, undersample no-falls a bit
        target_size = int((len(fall_windows) + len(no_fall_windows)) / 3)
        
        fall_oversampled = resample(
            fall_windows,
            replace=True,
            n_samples=target_size,
            random_state=seed_value
        )
        no_fall_undersampled = resample(
            no_fall_windows,
            replace=False,
            n_samples=target_size,
            random_state=seed_value
        )
        X = np.array(fall_oversampled + no_fall_undersampled)
        y = np.array([1]*target_size + [0]*target_size)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y


def build_lstm_model(input_shape=(48, 17, 3)):
    """
    Build LSTM model for fall detection
    
    Args:
        input_shape: (48, 17, 3) - [frames, keypoints, features]
    
    Returns:
        model
    """
    model = keras.Sequential([
        # Reshape: (48, 17, 3) -> (48, 51) - flatten keypoints per frame
        layers.Reshape((input_shape[0], input_shape[1] * input_shape[2]), 
                      input_shape=input_shape),
        
        # LSTM layers
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    return model


def plot_training_history(history, timestamp, accuracy_train_pos_y=20, accuracy_val_pos_y=-25, 
                          loss_train_pos_y=20, loss_val_pos_y=40, model_name="Model"):
    """
    Plot training and validation loss and accuracy curves with pins showing final values
    
    Parameters:
    history: training history object returned by model.fit()
    timestamp: datetime string for filename
    model_name: string name for the model (for title)
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get final epoch number and values
    final_epoch = len(history.history['loss']) - 1
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    # Plot Loss
    ax1.plot(history.history['loss'], label='train', color='blue', linewidth=2)
    ax1.plot(history.history['val_loss'], label='val', color='orange', linewidth=2)
    
    # Add pins and values for loss
    ax1.scatter(final_epoch, final_train_loss, color='blue', s=100, zorder=5, marker='o')
    ax1.scatter(final_epoch, final_val_loss, color='orange', s=100, zorder=5, marker='o')
    
    # Add text annotations for loss values
    ax1.annotate(f'{final_train_loss:.4f}', 
                xy=(final_epoch, final_train_loss), 
                xytext=(10, loss_train_pos_y), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7),
                color='white', fontsize=9, fontweight='bold')
    
    ax1.annotate(f'{final_val_loss:.4f}', 
                xy=(final_epoch, final_val_loss), 
                xytext=(10, loss_val_pos_y), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                color='white', fontsize=9, fontweight='bold')
    
    ax1.set_title('Loss (Training History)')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Accuracy  
    ax2.plot(history.history['accuracy'], label='train', color='blue', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='val', color='orange', linewidth=2)
    
    # Add pins and values for accuracy
    ax2.scatter(final_epoch, final_train_acc, color='blue', s=100, zorder=5, marker='o')
    ax2.scatter(final_epoch, final_val_acc, color='orange', s=100, zorder=5, marker='o')
    
    # Add text annotations for accuracy values
    ax2.annotate(f'{final_train_acc:.4f}', 
                xy=(final_epoch, final_train_acc), 
                xytext=(10, accuracy_train_pos_y), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7),
                color='white', fontsize=9, fontweight='bold')
    
    ax2.annotate(f'{final_val_acc:.4f}', 
                xy=(final_epoch, final_val_acc), 
                xytext=(10, accuracy_val_pos_y), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                color='white', fontsize=9, fontweight='bold')
    
    ax2.set_title('Accuracy (Training History)')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'{model_name} Training History', fontsize=14)
    
    plt.tight_layout()
    
    # Save figure
    save_path = f"./models/his_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {save_path}")
    
    plt.show()


def show_confusion_matrix(model, x_data, y_data, timestamp, class_names=None, figsize=(10, 8)):
    """
    Display confusion matrix with visualization
    Modified to handle binary labels (0s and 1s) instead of one-hot encoded
    
    Parameters:
    model: trained model
    x_data: input data
    y_data: true labels (binary: 0 or 1)
    timestamp: datetime string for filename
    class_names: list of class names for display
    figsize: size of the plot
    """
    # Get predictions
    y_pred = model.predict(x_data, verbose=0)
    
    # Convert predictions to binary (threshold at 0.5)
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()
    
    # y_data is already binary (0 or 1), use directly
    y_true = y_data.astype(int)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # Display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save figure
    save_path = f"./models/cf_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm


def main():
    print("="*60)
    print("FALL DETECTION LSTM TRAINING")
    print("="*60)
    
    # Generate timestamp for this training session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure models directory exists
    os.makedirs("./models", exist_ok=True)
    
    # Step 1: Load data
    print("\n[1/9] Loading pkl files...")
    all_persons = load_all_pkl_files(PKL_FOLDER)
    print(f"Loaded {len(all_persons)} persons")
    
    # Step 2: Forward fill
    print("\n[2/9] Applying forward-fill...")
    for person_data in all_persons:
        person_data['keypoints_filled'] = forward_fill_sequence(
            person_data['keypoints'],
            person_data['valid_frames'],
            max_fill=MAX_FORWARD_FILL
        )
    print("Forward-fill complete")
    
    # Step 3: Create windows
    print("\n[3/9] Creating windows...")
    all_fall_windows = []
    all_no_fall_windows = []
    
    for person_data in all_persons:
        windows, labels = create_windows(
            person_data['keypoints_filled'],
            person_data['frame_labels'],
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            min_fall_frames=MIN_FALL_FRAMES
        )
        
        for window, label in zip(windows, labels):
            if label == 'fall':
                all_fall_windows.append(window)
            else:
                all_no_fall_windows.append(window)
    
    print(f"Fall windows: {len(all_fall_windows)}")
    print(f"No-fall windows: {len(all_no_fall_windows)}")
    
    if len(all_fall_windows) == 0:
        print("ERROR: No fall windows found! Cannot train model.")
        return
    
    # Step 4: Balance dataset
    print("\n[4/9] Balancing dataset...")
    X, y = balance_dataset(all_fall_windows, all_no_fall_windows, strategy='hybrid')
    print(f"Balanced dataset shape: {X.shape}")
    print(f"Fall samples: {np.sum(y==1)}, No-fall samples: {np.sum(y==0)}")
    
    # Step 5: Split data
    print("\n[5/9] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed_value, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test set: {X_test.shape}, Labels: {y_test.shape}")
    
    # Step 6: Build model
    print("\n[6/9] Building model...")
    model = build_lstm_model(input_shape=(WINDOW_SIZE, 17, 3))
    model.summary()
    
    # Step 7: Train
    print("\n[7/9] Training model...")
    model_path = f"./models/fall_detection_{timestamp}.h5"
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ],
        verbose=1
    )
    
    print(f"\nModel saved to: {model_path}")
    
    # Step 8: Evaluate
    print("\n[8/9] Evaluating model...")
    test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print("="*60)
    
    # Step 9: Visualizations
    print("\n[9/9] Creating visualizations...")
    
    # Plot training history
    plot_training_history(
        history,
        timestamp,
        accuracy_train_pos_y=20,
        accuracy_val_pos_y=-25,
        loss_train_pos_y=20,
        loss_val_pos_y=40,
        model_name="Fall Detection LSTM"
    )
    
    # Show confusion matrix
    show_confusion_matrix(
        model,
        X_test,
        y_test,
        timestamp,
        class_names=['No Fall', 'Fall'],
        figsize=(10, 8)
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Model saved: ./models/fall_detection_{timestamp}.h5")
    print(f"Training history plot: ./models/his_{timestamp}.png")
    print(f"Confusion matrix: ./models/cf_{timestamp}.png")
    print("="*60)


if __name__ == "__main__":
    main()