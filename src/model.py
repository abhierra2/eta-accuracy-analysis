"""
PyTorch neural network model for ETA prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os


class ETADataset(Dataset):
    """
    PyTorch Dataset for ETA prediction.
    """
    
    def __init__(self, features, targets):
        """
        Args:
            features: numpy array of features
            targets: numpy array of target values (trip_duration_sec)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class ETAPredictor(nn.Module):
    """
    Neural network model for predicting trip duration (ETA).
    
    Architecture:
    - Input layer: feature_size
    - Hidden layers: 2-3 fully connected layers with ReLU activation
    - Output layer: single value (predicted duration in seconds)
    """
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.2):
        """
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super(ETAPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer (no activation, as we're predicting continuous values)
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Predicted duration in seconds
        """
        return self.network(x).squeeze()


def prepare_features(df, scaler=None, fit_scaler=False):
    """
    Prepare features from trip data for model training/prediction.
    
    Args:
        df: DataFrame with trip data
        scaler: Optional pre-fitted StandardScaler (for prediction)
        fit_scaler: If True and scaler is None, fit a new scaler
    
    Returns:
        Tuple of (features_array, feature_names, scaler)
    """
    # Create feature DataFrame with same index as input
    features_df = pd.DataFrame(index=df.index)
    
    # Numerical features
    features_df['haversine_distance_km'] = df['haversine_distance_km']
    features_df['hour_of_day'] = df['hour_of_day']
    features_df['day_of_week'] = df['day_of_week']
    
    # Cyclical encoding for hour (sin/cos)
    features_df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    features_df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    
    # Cyclical encoding for day of week
    features_df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    features_df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # One-hot encode distance bucket
    # Get all possible distance buckets to ensure consistent columns
    all_buckets = ['<1mi', '1-3mi', '3-5mi', '5-10mi', '10+mi']
    distance_dummies = pd.get_dummies(df['distance_bucket'], prefix='dist')
    # Ensure all buckets are present
    for bucket in all_buckets:
        col_name = f'dist_{bucket}'
        if col_name not in distance_dummies.columns:
            distance_dummies[col_name] = 0
    # Reorder columns for consistency
    distance_dummies = distance_dummies[[f'dist_{b}' for b in all_buckets]]
    # Ensure index alignment
    distance_dummies.index = df.index
    features_df = pd.concat([features_df, distance_dummies], axis=1)
    
    # Optional: Add coordinate features (normalized)
    if 'pickup_latitude' in df.columns:
        # Normalize coordinates to NYC area (roughly)
        features_df['pickup_lat_norm'] = (df['pickup_latitude'] - 40.7) / 0.5
        features_df['pickup_lon_norm'] = (df['pickup_longitude'] + 74.0) / 0.5
        features_df['dropoff_lat_norm'] = (df['dropoff_latitude'] - 40.7) / 0.5
        features_df['dropoff_lon_norm'] = (df['dropoff_longitude'] + 74.0) / 0.5
    
    # Ensure we have the same number of rows as input
    assert len(features_df) == len(df), f"Feature preparation changed row count: {len(features_df)} != {len(df)}"
    
    # Get feature names
    feature_names = features_df.columns.tolist()
    
    # Convert to numpy array
    features_array = features_df.values.astype(np.float32)
    
    # Scale features
    if scaler is not None:
        # Use provided scaler (for prediction)
        features_scaled = scaler.transform(features_array)
    elif fit_scaler:
        # Fit new scaler (for training)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)
    else:
        # No scaling
        features_scaled = features_array
        scaler = None
    
    return features_scaled, feature_names, scaler


def train_model(train_df, val_df=None, config=None):
    """
    Train the ETA prediction model.
    
    Args:
        train_df: Training DataFrame with trip data
        val_df: Optional validation DataFrame
        config: Dictionary with training configuration
    
    Returns:
        Tuple of (trained_model, scaler, training_history)
    """
    if config is None:
        config = {
            'batch_size': 256,
            'learning_rate': 0.001,
            'num_epochs': 50,
            'hidden_sizes': [128, 64, 32],
            'dropout_rate': 0.2,
            'early_stopping_patience': 10,
            'lr_scheduler_factor': 0.5,  # Reduce LR by 50% when plateau
            'lr_scheduler_patience': 5,  # Wait 5 epochs before reducing LR
            'min_learning_rate': 1e-6    # Minimum learning rate
        }
    
    print("\n" + "="*60)
    print("PREPARING FEATURES")
    print("="*60)
    
    # Prepare features
    X_train, feature_names, scaler = prepare_features(train_df, fit_scaler=True)
    y_train = train_df['trip_duration_sec'].values.astype(np.float32)
    
    # Ensure X_train and y_train have the same length
    min_len = min(len(X_train), len(y_train))
    if len(X_train) != len(y_train):
        print(f"Warning: Feature and target lengths don't match: {len(X_train)} vs {len(y_train)}")
        print(f"Truncating to minimum length: {min_len}")
        X_train = X_train[:min_len]
        y_train = y_train[:min_len]
    
    print(f"Feature count: {len(feature_names)}")
    print(f"Training samples: {len(X_train):,}")
    
    # Create datasets
    train_dataset = ETADataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    
    val_loader = None
    if val_df is not None:
        X_val, _, _ = prepare_features(val_df, scaler=scaler)
        y_val = val_df['trip_duration_sec'].values.astype(np.float32)
        
        # Ensure X_val and y_val have the same length
        min_len = min(len(X_val), len(y_val))
        if len(X_val) != len(y_val):
            print(f"Warning: Feature and target lengths don't match: {len(X_val)} vs {len(y_val)}")
            print(f"Truncating to minimum length: {min_len}")
            X_val = X_val[:min_len]
            y_val = y_val[:min_len]
        
        val_dataset = ETADataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        print(f"Validation samples: {len(X_val):,}")
    
    # Initialize model
    input_size = X_train.shape[1]
    model = ETAPredictor(
        input_size=input_size,
        hidden_sizes=config['hidden_sizes'],
        dropout_rate=config['dropout_rate']
    )
    
    # Loss and optimizer
    criterion = nn.L1Loss()  # MAE loss
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler - reduces LR when validation loss plateaus
    scheduler = None
    if val_df is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('lr_scheduler_factor', 0.5),
            patience=config.get('lr_scheduler_patience', 5),
            min_lr=config.get('min_learning_rate', 1e-6)
        )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    # Track best model for early stopping
    best_model_state = None
    
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    print(f"Model architecture: {input_size} -> {config['hidden_sizes']} -> 1")
    print(f"Batch size: {config['batch_size']}")
    print(f"Initial learning rate: {config['learning_rate']}")
    print(f"Epochs: {config['num_epochs']}")
    if val_df is not None:
        print(f"Early stopping patience: {config['early_stopping_patience']}")
        print(f"LR scheduler: ReduceLROnPlateau (factor={config.get('lr_scheduler_factor', 0.5)}, "
              f"patience={config.get('lr_scheduler_patience', 5)})")
    print("="*60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for features, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for features, targets in val_loader:
                    predictions = model(features)
                    loss = criterion(predictions, targets)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            history['val_loss'].append(avg_val_loss)
            
            # Learning rate scheduling
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)
            
            if scheduler is not None:
                scheduler.step(avg_val_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != current_lr:
                    print(f"  Learning rate reduced to {new_lr:.2e}")
            
            # Early stopping with best model checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= config['early_stopping_patience']:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    print(f"Best validation loss: {best_val_loss:.2f} (at epoch {epoch + 1 - patience_counter})")
                    # Restore best model
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                        print("Restored best model weights")
                    break
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{config['num_epochs']}: "
                      f"Train Loss: {avg_train_loss:.2f}, "
                      f"Val Loss: {avg_val_loss:.2f}, "
                      f"LR: {current_lr:.2e}")
        else:
            # Track learning rate even without validation
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{config['num_epochs']}: "
                      f"Train Loss: {avg_train_loss:.2f}, "
                      f"LR: {current_lr:.2e}")
    
    print(f"\nTraining complete. Final train loss: {avg_train_loss:.2f}")
    if val_loader is not None:
        print(f"Final validation loss: {avg_val_loss:.2f}")
        print(f"Best validation loss: {best_val_loss:.2f}")
        # Ensure we're using the best model
        if best_model_state is not None and avg_val_loss > best_val_loss:
            model.load_state_dict(best_model_state)
            print("Restored best model weights")
    
    return model, scaler, history


def predict(model, scaler, df):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained PyTorch model
        scaler: Fitted StandardScaler
        df: DataFrame with trip data
    
    Returns:
        numpy array of predictions
    """
    model.eval()
    
    # Prepare features (using the same scaler)
    X, _, _ = prepare_features(df, scaler=scaler)
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X_tensor).numpy()
    
    # Ensure non-negative predictions
    predictions = np.maximum(predictions, 30.0)  # Minimum 30 seconds
    
    return predictions


def save_model(model, scaler, feature_names, filepath):
    """
    Save model and scaler to disk.
    
    Args:
        model: Trained PyTorch model
        scaler: Fitted StandardScaler
        feature_names: List of feature names
        filepath: Path to save model (without extension)
    """
    # Save model state
    torch.save(model.state_dict(), f"{filepath}.pth")
    
    # Save scaler and feature names
    with open(f"{filepath}_scaler.pkl", 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'feature_names': feature_names
        }, f)
    
    print(f"Model saved to {filepath}.pth")
    print(f"Scaler saved to {filepath}_scaler.pkl")


def load_model(filepath, input_size=None, hidden_sizes=[128, 64, 32], dropout_rate=0.2):
    """
    Load model and scaler from disk.
    
    Args:
        filepath: Path to model files (without extension)
        input_size: Input size (will be inferred from scaler if not provided)
        hidden_sizes: Hidden layer sizes
        dropout_rate: Dropout rate
    
    Returns:
        Tuple of (model, scaler, feature_names)
    """
    # Load scaler and feature names
    with open(f"{filepath}_scaler.pkl", 'rb') as f:
        data = pickle.load(f)
        scaler = data['scaler']
        feature_names = data['feature_names']
    
    # Infer input size from scaler
    if input_size is None:
        input_size = scaler.n_features_in_
    
    # Initialize model
    model = ETAPredictor(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate
    )
    
    # Load model weights
    model.load_state_dict(torch.load(f"{filepath}.pth"))
    model.eval()
    
    print(f"Model loaded from {filepath}.pth")
    
    return model, scaler, feature_names

