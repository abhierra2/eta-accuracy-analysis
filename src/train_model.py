"""
Train the PyTorch ETA prediction model.
"""

import pandas as pd
from sqlalchemy import create_engine, text
import sys
import os
import argparse

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DB_CONNECTION_STRING, METRICS_DIR
from model import train_model, save_model, prepare_features


def load_training_data(engine):
    """
    Load training data from database.
    
    Args:
        engine: SQLAlchemy engine
    
    Returns:
        DataFrame with training data
    """
    query = """
    SELECT 
        trip_id,
        trip_duration_sec,
        haversine_distance_km,
        hour_of_day,
        day_of_week,
        distance_bucket,
        pickup_latitude,
        pickup_longitude,
        dropoff_latitude,
        dropoff_longitude
    FROM trips_training
    WHERE trip_duration_sec > 0 
      AND haversine_distance_km > 0
      AND hour_of_day IS NOT NULL
      AND day_of_week IS NOT NULL
      AND distance_bucket IS NOT NULL
    ORDER BY pickup_datetime
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    return df


def main():
    """
    Main function to train the model.
    """
    parser = argparse.ArgumentParser(description='Train ETA prediction model')
    parser.add_argument(
        '--model-dir',
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'models'),
        help='Directory to save model files'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.2,
        help='Validation split ratio (from training data)'
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=10,
        help='Early stopping patience (epochs without improvement)'
    )
    parser.add_argument(
        '--lr-scheduler-factor',
        type=float,
        default=0.5,
        help='Learning rate reduction factor (default: 0.5)'
    )
    parser.add_argument(
        '--lr-scheduler-patience',
        type=int,
        default=5,
        help='Learning rate scheduler patience (epochs before reducing LR)'
    )
    parser.add_argument(
        '--min-learning-rate',
        type=float,
        default=1e-6,
        help='Minimum learning rate (default: 1e-6)'
    )
    
    args = parser.parse_args()
    
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Create database connection
    print("Connecting to Postgres database...")
    engine = create_engine(DB_CONNECTION_STRING)
    
    # Test connection
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM trips_training"))
            count = result.scalar()
            print(f"Connected. Found {count:,} trips in training set")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("Please ensure:")
        print("  1. Postgres is running")
        print("  2. Database 'eta_analysis' exists")
        print("  3. You've run load_to_postgres.py and 01_create_tables.sql")
        sys.exit(1)
    
    # Load training data
    print("\nLoading training data...")
    train_df = load_training_data(engine)
    print(f"Loaded {len(train_df):,} training trips")
    
    # Split into train/val if requested
    val_df = None
    if args.val_split > 0:
        split_idx = int(len(train_df) * (1 - args.val_split))
        val_df = train_df.iloc[split_idx:].copy()
        train_df = train_df.iloc[:split_idx].copy()
        print(f"Split: {len(train_df):,} train, {len(val_df):,} validation")
    
    # Training configuration
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.epochs,
        'hidden_sizes': [128, 64, 32],
        'dropout_rate': 0.2,
        'early_stopping_patience': args.early_stopping_patience,
        'lr_scheduler_factor': args.lr_scheduler_factor,
        'lr_scheduler_patience': args.lr_scheduler_patience,
        'min_learning_rate': args.min_learning_rate
    }
    
    # Train model
    model, scaler, history = train_model(train_df, val_df, config)
    
    # Get feature names by preparing features on a sample
    _, feature_names, _ = prepare_features(train_df.head(1), scaler=scaler)
    
    # Save model
    model_path = os.path.join(args.model_dir, 'eta_model')
    save_model(model, scaler, feature_names, model_path)
    
    print("\nModel training complete!")
    print(f"Model saved to: {model_path}")


if __name__ == '__main__':
    main()

