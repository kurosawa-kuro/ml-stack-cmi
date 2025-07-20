#!/usr/bin/env python3
"""
CMI Competition - 1D CNN Training Pipeline
==========================================
Week 2: 1D CNN implementation for multimodal sensor data

Features:
- InceptionTime-inspired 1D CNN architecture
- Multimodal sensor fusion (IMU, ToF, Thermopile)
- Data augmentation for time-series
- GroupKFold validation (participant-aware)
- Mixed precision training
- Comprehensive evaluation and submission
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, classification_report
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def setup_cnn_training():
    """Setup CNN training environment"""
    output_dir = Path(__file__).parent.parent / "outputs" / "models" / "cnn_1d"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure TensorFlow for mixed precision
    tf.config.optimizer.set_jit(True)
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    print("🔧 TensorFlow GPU configuration:")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  ✓ Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"  ⚠️  GPU configuration error: {e}")
    else:
        print("  ℹ️  Running on CPU")
    
    return output_dir

def load_sequence_data():
    """Load time-series sequence data for CNN training"""
    print("📊 Loading sequence data for CNN training...")
    
    try:
        from src.data.gold import get_sequence_data  # Assume this function exists
        X_train, y_train, X_test, groups = get_sequence_data()
        
        print(f"  ✓ Train sequences: {X_train.shape}")
        print(f"  ✓ Train labels: {y_train.shape}")  
        print(f"  ✓ Test sequences: {X_test.shape}")
        print(f"  ✓ Groups (participants): {len(np.unique(groups))}")
        
        return X_train, y_train, X_test, groups
        
    except ImportError:
        print("  ⚠️  get_sequence_data not implemented. Creating dummy data...")
        # Create dummy data for demonstration
        n_train = 10000
        n_test = 5000
        seq_length = 100  # 2 seconds at 50Hz
        n_features = 12   # 3 acc + 3 gyro + 4 tof + 2 thermopile (simplified)
        n_classes = 5
        
        X_train = np.random.randn(n_train, seq_length, n_features).astype(np.float32)
        y_train = np.random.randint(0, n_classes, n_train)
        X_test = np.random.randn(n_test, seq_length, n_features).astype(np.float32)
        groups = np.random.randint(0, 100, n_train)  # 100 participants
        
        print(f"  ⚠️  Using dummy data: train={X_train.shape}, test={X_test.shape}")
        return X_train, y_train, X_test, groups
        
    except Exception as e:
        print(f"  ✗ Failed to load sequence data: {e}")
        return None, None, None, None

def create_inception_block(x, filters, kernel_sizes=[1, 3, 5], name="inception"):
    """Create an Inception-style block for 1D CNN"""
    branches = []
    
    for i, kernel_size in enumerate(kernel_sizes):
        branch = layers.Conv1D(
            filters // len(kernel_sizes), 
            kernel_size, 
            padding='same',
            activation='relu',
            name=f"{name}_conv_{kernel_size}"
        )(x)
        branches.append(branch)
    
    # MaxPooling branch
    pool_branch = layers.MaxPooling1D(3, strides=1, padding='same', name=f"{name}_pool")(x)
    pool_branch = layers.Conv1D(
        filters // len(kernel_sizes), 
        1, 
        padding='same',
        activation='relu',
        name=f"{name}_pool_conv"
    )(pool_branch)
    branches.append(pool_branch)
    
    # Concatenate all branches
    output = layers.Concatenate(name=f"{name}_concat")(branches)
    return output

def create_multimodal_cnn(input_shape, n_classes, architecture="inception"):
    """Create multimodal 1D CNN architecture"""
    print(f"🏗️  Building {architecture} CNN architecture...")
    
    # Input layer
    inputs = layers.Input(shape=input_shape, name="sensor_input")
    
    if architecture == "inception":
        # InceptionTime-inspired architecture
        x = inputs
        
        # Initial convolution
        x = layers.Conv1D(64, 7, padding='same', activation='relu', name="initial_conv")(x)
        x = layers.BatchNormalization(name="initial_bn")(x)
        
        # Inception blocks
        x = create_inception_block(x, 128, name="inception_1")
        x = layers.BatchNormalization(name="bn_1")(x)
        x = layers.Dropout(0.2, name="dropout_1")(x)
        
        x = create_inception_block(x, 256, name="inception_2")
        x = layers.BatchNormalization(name="bn_2")(x)
        x = layers.Dropout(0.2, name="dropout_2")(x)
        
        x = create_inception_block(x, 512, name="inception_3")
        x = layers.BatchNormalization(name="bn_3")(x)
        x = layers.Dropout(0.3, name="dropout_3")(x)
        
    else:  # Simple CNN architecture
        x = inputs
        
        # Convolutional layers
        for i, filters in enumerate([64, 128, 256]):
            x = layers.Conv1D(filters, 3, padding='same', activation='relu', name=f"conv_{i+1}")(x)
            x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
            x = layers.MaxPooling1D(2, name=f"pool_{i+1}")(x)
            x = layers.Dropout(0.2, name=f"dropout_{i+1}")(x)
    
    # Global pooling and classification head
    x = layers.GlobalAveragePooling1D(name="global_pool")(x)
    x = layers.Dense(512, activation='relu', name="dense_1")(x)
    x = layers.Dropout(0.5, name="final_dropout")(x)
    
    # Output layer
    if n_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid', name="output", dtype='float32')(x)
    else:
        outputs = layers.Dense(n_classes, activation='softmax', name="output", dtype='float32')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name=f"cnn_1d_{architecture}")
    
    print(f"  ✓ Model created with {model.count_params():,} parameters")
    return model

def create_data_augmentation():
    """Create data augmentation pipeline for time-series"""
    def augment_sequence(x, y):
        # Time shifting (random crop)
        shift = tf.random.uniform([], -5, 6, dtype=tf.int32)
        x = tf.roll(x, shift, axis=0)
        
        # Gaussian noise injection
        noise_factor = 0.01
        noise = tf.random.normal(tf.shape(x), stddev=noise_factor)
        x = x + noise
        
        # Speed variation (simple time dilation/compression)
        if tf.random.uniform([]) < 0.3:  # 30% chance
            # Simple approach: subsample or repeat samples
            indices = tf.range(tf.shape(x)[0])
            if tf.random.uniform([]) < 0.5:
                # Speed up (subsample)
                indices = indices[::2]
            else:
                # Slow down (repeat samples)
                indices = tf.repeat(indices, 2)[:tf.shape(x)[0]]
            
            x = tf.gather(x, indices)
        
        return x, y
    
    return augment_sequence

def evaluate_composite_f1_tf(y_true, y_pred, n_classes):
    """Calculate composite F1 score for TensorFlow predictions"""
    # Convert predictions to class labels
    if n_classes == 2:
        y_pred_labels = (y_pred > 0.5).astype(int).flatten()
    else:
        y_pred_labels = np.argmax(y_pred, axis=1)
    
    # Binary classification (behavior vs no behavior)
    y_true_binary = (y_true > 0).astype(int)
    y_pred_binary = (y_pred_labels > 0).astype(int)
    
    binary_f1 = f1_score(y_true_binary, y_pred_binary, average='binary')
    
    # Multiclass classification
    macro_f1 = f1_score(y_true, y_pred_labels, average='macro')
    
    # Composite score
    composite_f1 = (binary_f1 + macro_f1) / 2
    
    return composite_f1, binary_f1, macro_f1

def train_cnn_with_cv(X_train, y_train, groups, output_dir, n_splits=5):
    """Train CNN with GroupKFold cross-validation"""
    print(f"\n🚀 Training 1D CNN with {n_splits}-fold GroupKFold CV...")
    
    # Prepare label encoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    n_classes = len(le.classes_)
    
    print(f"  Number of classes: {n_classes}")
    print(f"  Classes: {le.classes_}")
    
    # Initialize GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    
    # Track metrics across folds
    fold_metrics = []
    models = []
    oof_predictions = np.zeros((len(y_train), n_classes if n_classes > 2 else 1))
    
    # Data augmentation
    augment_fn = create_data_augmentation()
    
    # Cross-validation training
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_encoded, groups)):
        print(f"\n📊 Training Fold {fold + 1}/{n_splits}")
        
        # Split data
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_encoded[train_idx], y_encoded[val_idx]
        
        # Create model
        model = create_multimodal_cnn(X_train.shape[1:], n_classes)
        
        # Compile model
        if n_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_fold_train, y_fold_train))
        train_dataset = (train_dataset
                        .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
                        .batch(32)
                        .prefetch(tf.data.AUTOTUNE))
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_fold_val, y_fold_val))
        val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        
        # Callbacks
        model_callbacks = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=100,
            callbacks=model_callbacks,
            verbose=1
        )
        
        # Predict on validation set
        val_pred = model.predict(X_fold_val, verbose=0)
        
        # Store out-of-fold predictions
        oof_predictions[val_idx] = val_pred
        
        # Calculate metrics
        composite_f1, binary_f1, macro_f1 = evaluate_composite_f1_tf(y_fold_val, val_pred, n_classes)
        
        fold_metrics.append({
            'fold': fold + 1,
            'composite_f1': composite_f1,
            'binary_f1': binary_f1,
            'macro_f1': macro_f1,
            'val_loss': min(history.history['val_loss']),
            'val_accuracy': max(history.history['val_accuracy'])
        })
        
        print(f"  Fold {fold + 1} - Composite F1: {composite_f1:.4f} (Binary: {binary_f1:.4f}, Macro: {macro_f1:.4f})")
        
        # Save model
        models.append(model)
        model_path = output_dir / f"cnn_fold_{fold + 1}.h5"
        model.save(str(model_path))
    
    # Calculate overall CV performance
    if n_classes == 2:
        oof_labels = (oof_predictions > 0.5).astype(int).flatten()
    else:
        oof_labels = np.argmax(oof_predictions, axis=1)
    
    overall_composite, overall_binary, overall_macro = evaluate_composite_f1_tf(y_encoded, oof_predictions, n_classes)
    
    cv_results = {
        'cv_composite_f1': overall_composite,
        'cv_binary_f1': overall_binary,
        'cv_macro_f1': overall_macro,
        'cv_std': np.std([m['composite_f1'] for m in fold_metrics]),
        'fold_metrics': fold_metrics,
        'n_splits': n_splits,
        'n_classes': n_classes
    }
    
    print(f"\n📈 Overall CV Results:")
    print(f"  Composite F1: {overall_composite:.4f} ± {cv_results['cv_std']:.4f}")
    print(f"  Binary F1: {overall_binary:.4f}")
    print(f"  Macro F1: {overall_macro:.4f}")
    
    # Save CV results
    with open(output_dir / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)
    
    # Save out-of-fold predictions
    oof_df = pd.DataFrame({
        'true_label': y_train,
        'oof_prediction': oof_labels
    })
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)
    
    # Save label encoder
    import joblib
    joblib.dump(le, output_dir / "label_encoder.pkl")
    
    return models, cv_results, oof_predictions, le

def generate_test_predictions(models, X_test, le, output_dir):
    """Generate test predictions using ensemble of CNN models"""
    print("\n🔮 Generating test predictions...")
    
    # Average predictions across all folds
    test_predictions = np.zeros((len(X_test), len(models[0].predict(X_test[:1], verbose=0)[0])))
    
    for i, model in enumerate(models):
        pred = model.predict(X_test, verbose=0)
        test_predictions += pred
        
    test_predictions /= len(models)
    
    # Convert to class labels
    if len(test_predictions.shape) > 1 and test_predictions.shape[1] > 1:
        test_pred_labels = np.argmax(test_predictions, axis=1)
    else:
        test_pred_labels = (test_predictions > 0.5).astype(int).flatten()
    
    # Convert back to original labels
    test_pred_original = le.inverse_transform(test_pred_labels)
    
    # Create submission format
    submission = pd.DataFrame({
        'id': range(len(test_pred_original)),
        'label': test_pred_original
    })
    
    submission_path = output_dir / f"submission_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    submission.to_csv(submission_path, index=False)
    
    print(f"  ✓ Submission saved to: {submission_path}")
    
    return submission, test_pred_original

def generate_cnn_training_report(cv_results, output_dir):
    """Generate comprehensive CNN training report"""
    print("\n📋 Generating CNN training report...")
    
    report_content = f"""
# 1D CNN Training Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Performance
- **Composite F1 Score**: {cv_results['cv_composite_f1']:.4f} ± {cv_results['cv_std']:.4f}
- **Binary F1 Score**: {cv_results['cv_binary_f1']:.4f}
- **Macro F1 Score**: {cv_results['cv_macro_f1']:.4f}

## Model Architecture
- **Type**: 1D CNN with Inception-inspired blocks
- **Input Shape**: Time-series sequences
- **Number of Classes**: {cv_results['n_classes']}
- **Training Strategy**: GroupKFold cross-validation

## Cross-Validation Setup
- **Strategy**: GroupKFold (participant-aware)
- **Number of Folds**: {cv_results['n_splits']}
- **Data Augmentation**: Time shifting, Gaussian noise, speed variation

## Fold-by-Fold Results
"""
    
    for fold_metric in cv_results['fold_metrics']:
        report_content += f"- **Fold {fold_metric['fold']}**: {fold_metric['composite_f1']:.4f} (Binary: {fold_metric['binary_f1']:.4f}, Macro: {fold_metric['macro_f1']:.4f})\n"
    
    report_content += f"""

## Training Configuration
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: {'Binary crossentropy' if cv_results['n_classes'] == 2 else 'Sparse categorical crossentropy'}
- **Batch Size**: 32
- **Early Stopping**: Patience 10 on validation loss
- **Learning Rate Reduction**: Factor 0.5, patience 5

## Data Augmentation
- **Time Shifting**: Random temporal shifts (±5 samples)
- **Gaussian Noise**: 1% noise injection
- **Speed Variation**: Random time dilation/compression (30% probability)

## Files Generated
- `cnn_fold_*.h5` - Trained CNN models for each fold
- `cv_results.json` - Cross-validation metrics
- `oof_predictions.csv` - Out-of-fold predictions
- `label_encoder.pkl` - Label encoder for prediction conversion
- `submission_cnn_*.csv` - Test predictions for submission

## Next Steps
1. **Error Analysis**: Run `python scripts/workflow-to-evaluate.py`
2. **Ensemble**: Combine with LightGBM using `python scripts/workflow-to-training-ensemble.py`
3. **Hyperparameter Tuning**: Optimize architecture and training parameters
4. **Advanced Augmentation**: Implement more sophisticated time-series augmentations

## Competition Context
- **Target Score**: Bronze medal (LB 0.60+)
- **Current Model**: 1D CNN with multimodal sensor fusion
- **Next Phase**: Ensemble methods and advanced optimization
"""

    with open(output_dir / "CNN_TRAINING_REPORT.md", "w") as f:
        f.write(report_content)
    
    print(f"  ✓ CNN training report saved to: {output_dir}/CNN_TRAINING_REPORT.md")

def main():
    """Main CNN training workflow"""
    print("🧠 CMI Competition - 1D CNN Training Pipeline")
    print("=" * 50)
    
    # Setup
    output_dir = setup_cnn_training()
    
    # Load sequence data
    X_train, y_train, X_test, groups = load_sequence_data()
    if X_train is None:
        print("❌ Failed to load sequence data.")
        print("Please ensure the gold layer can provide sequence data:")
        print("  1. python scripts/workflow-to-bronze.py")
        print("  2. python scripts/workflow-to-silver.py")
        print("  3. python scripts/workflow-to-gold.py")
        sys.exit(1)
    
    # Train CNN with cross-validation
    models, cv_results, oof_predictions, le = train_cnn_with_cv(
        X_train, y_train, groups, output_dir
    )
    
    # Generate test predictions
    submission, test_pred_labels = generate_test_predictions(models, X_test, le, output_dir)
    
    # Generate comprehensive report
    generate_cnn_training_report(cv_results, output_dir)
    
    # Final summary
    print("\n" + "=" * 50)
    print("✅ 1D CNN training completed!")
    print(f"📊 Final CV Score: {cv_results['cv_composite_f1']:.4f} ± {cv_results['cv_std']:.4f}")
    print(f"📁 Results saved to: {output_dir}")
    
    # Competition context
    target_score = 0.60
    current_score = cv_results['cv_composite_f1']
    
    if current_score >= target_score:
        print(f"🎉 Excellent! CV score {current_score:.4f} exceeds bronze target {target_score}")
    else:
        gap = target_score - current_score
        print(f"📈 CV score {current_score:.4f} is {gap:.4f} below bronze target {target_score}")
    
    print("\n🚀 Next steps:")
    print("  1. Review CNN_TRAINING_REPORT.md")
    print("  2. python scripts/workflow-to-evaluate.py          # Compare with LightGBM")
    print("  3. python scripts/workflow-to-training-ensemble.py # Create ensemble")
    print("  4. Submit to Kaggle to validate CV-LB alignment")

if __name__ == "__main__":
    main()