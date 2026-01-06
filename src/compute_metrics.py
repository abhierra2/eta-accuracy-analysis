"""
Compute ETA accuracy metrics using PyTorch model predictions and save results to CSV.
Supports Spark for loading large datasets efficiently.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DB_CONNECTION_STRING, METRICS_DIR
from model import load_model, predict

# Try to import Spark
try:
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


def load_evaluation_data(engine, use_spark=False, spark=None):
    """
    Load evaluation data from database.
    
    Args:
        engine: SQLAlchemy engine
        use_spark: Whether to use Spark for loading (faster for large datasets)
        spark: Optional SparkSession (required if use_spark=True)
    
    Returns:
        DataFrame with evaluation data (Pandas DataFrame)
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
        dropoff_longitude,
        pickup_datetime
    FROM trips_evaluation
    WHERE trip_duration_sec > 0 
      AND haversine_distance_km > 0
      AND hour_of_day IS NOT NULL
      AND day_of_week IS NOT NULL
      AND distance_bucket IS NOT NULL
    ORDER BY pickup_datetime
    """
    
    if use_spark and spark is not None:
        # Use Spark to read from Postgres (faster for large datasets)
        print("Loading evaluation data using Spark...")
        from urllib.parse import urlparse
        # Handle postgresql:// URL format
        conn_str = DB_CONNECTION_STRING
        if conn_str.startswith("postgresql://"):
            conn_str = conn_str.replace("postgresql://", "http://")
        parsed = urlparse(conn_str)
        
        # Extract components
        hostname = parsed.hostname or "localhost"
        port = parsed.port or 5432
        database = parsed.path.lstrip('/') if parsed.path else "eta_analysis"
        username = parsed.username or (engine.url.username if engine else None)
        password = parsed.password or (engine.url.password if engine else None)
        
        jdbc_url = f"jdbc:postgresql://{hostname}:{port}/{database}"
        properties = {
            "user": username,
            "password": password,
            "driver": "org.postgresql.Driver"
        }
        
        # Read using Spark JDBC
        spark_df = spark.read \
            .format("jdbc") \
            .option("url", jdbc_url) \
            .option("query", query) \
            .option("user", properties["user"]) \
            .option("password", properties["password"]) \
            .option("driver", properties["driver"]) \
            .load()
        
        # Convert to Pandas (PyTorch model expects Pandas/NumPy)
        print("Converting Spark DataFrame to Pandas...")
        df = spark_df.toPandas()
        print(f"Loaded {len(df):,} rows")
    else:
        # Use standard SQLAlchemy
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        print(f"Loaded {len(df):,} rows")
    
    return df


def store_predictions(engine, df, use_spark=False, spark=None):
    """
    Store predictions in database by creating/updating a table.
    Uses Spark JDBC for faster writes when available.
    
    Args:
        engine: SQLAlchemy engine
        df: DataFrame with predictions (must include estimated_duration_sec)
        use_spark: Whether to use Spark for writing (faster for large datasets)
        spark: Optional SparkSession
    """
    # Create temporary table with predictions
    df_to_store = df[['trip_id', 'estimated_duration_sec']].copy()
    row_count = len(df_to_store)
    
    # Create or replace predictions table
    with engine.begin() as conn:
        # Drop table if it exists to recreate with trip_id
        conn.execute(text("DROP TABLE IF EXISTS trip_predictions"))
        
        # Create table with trip_id as primary key
        conn.execute(text("""
            CREATE TABLE trip_predictions (
                trip_id BIGINT,
                estimated_duration_sec FLOAT,
                PRIMARY KEY (trip_id)
            )
        """))
        
        # Create index for fast joins (though primary key already provides this)
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_trip_predictions_trip_id 
            ON trip_predictions(trip_id)
        """))
    
    # Use Spark JDBC for large datasets if available
    if use_spark and spark is not None:
        try:
            print(f"Storing {row_count:,} predictions using Spark JDBC...")
            from urllib.parse import urlparse
            conn_str = DB_CONNECTION_STRING
            if conn_str.startswith("postgresql://"):
                conn_str = conn_str.replace("postgresql://", "http://")
            parsed = urlparse(conn_str)
            
            hostname = parsed.hostname or "localhost"
            port = parsed.port or 5432
            database = parsed.path.lstrip('/') if parsed.path else "eta_analysis"
            username = parsed.username or (engine.url.username if engine else None)
            password = parsed.password or (engine.url.password if engine else None)
            
            jdbc_url = f"jdbc:postgresql://{hostname}:{port}/{database}"
            
            # Convert to Spark DataFrame and write
            spark_df = spark.createDataFrame(df_to_store)
            spark_df.write \
                .format("jdbc") \
                .option("url", jdbc_url) \
                .option("dbtable", "trip_predictions") \
                .option("user", username) \
                .option("password", password) \
                .option("driver", "org.postgresql.Driver") \
                .mode("append") \
                .save()
            
            print(f"Successfully stored {row_count:,} predictions using Spark JDBC")
            return
        except Exception as e:
            print(f"Note: Spark JDBC write failed ({e}), falling back to Pandas method")
    
    # Fallback to Pandas method (slower but works)
    print(f"Storing {row_count:,} predictions using Pandas to_sql...")
    
    # Check for duplicates (shouldn't happen with trip_id, but just in case)
    duplicates = df_to_store.duplicated(subset=['trip_id']).sum()
    if duplicates > 0:
        print(f"  Warning: Found {duplicates:,} duplicate trip_id values. Deduplicating...")
        # For duplicates, keep the last value
        df_to_store = df_to_store.drop_duplicates(subset=['trip_id'], keep='last')
        print(f"  Deduplicated to {len(df_to_store):,} unique predictions")
    
    # Use standard insert (duplicates already removed)
    df_to_store.to_sql(
        'trip_predictions',
        engine,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=10000
    )
    print(f"Successfully stored {len(df_to_store):,} predictions using Pandas")


def create_metrics_views(engine):
    """
    Create SQL views for computing metrics from predictions.
    
    Args:
        engine: SQLAlchemy engine
    """
    sql = """
    -- Create view with predictions joined to evaluation data
    CREATE OR REPLACE VIEW trips_with_eta AS
    SELECT 
        e.*,
        p.estimated_duration_sec
    FROM trips_evaluation e
    INNER JOIN trip_predictions p
        ON e.trip_id = p.trip_id;
    
    -- Compute error metrics
    CREATE OR REPLACE VIEW eta_errors AS
    SELECT 
        *,
        estimated_duration_sec - trip_duration_sec AS error_sec,
        ABS(estimated_duration_sec - trip_duration_sec) AS abs_error_sec,
        (estimated_duration_sec - trip_duration_sec) / NULLIF(trip_duration_sec, 0) AS pct_error,
        ABS(estimated_duration_sec - trip_duration_sec) / NULLIF(trip_duration_sec, 0) AS abs_pct_error
    FROM trips_with_eta
    WHERE estimated_duration_sec IS NOT NULL 
      AND trip_duration_sec > 0;
    
    -- Aggregate metrics overall
    CREATE OR REPLACE VIEW metrics_overall AS
    SELECT 
        'overall' AS segment_type,
        'all' AS segment_value,
        COUNT(*) AS trip_count,
        AVG(abs_error_sec) AS mae_sec,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY abs_error_sec) AS medae_sec,
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY abs_error_sec) AS p90_abs_error_sec,
        AVG(abs_pct_error) AS mae_pct,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY abs_pct_error) AS medae_pct,
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY abs_pct_error) AS p90_abs_error_pct,
        AVG(CASE WHEN abs_pct_error <= 0.10 THEN 1.0 ELSE 0.0 END) AS pct_within_10pct,
        AVG(CASE WHEN abs_pct_error <= 0.20 THEN 1.0 ELSE 0.0 END) AS pct_within_20pct
    FROM eta_errors;
    
    -- Aggregate metrics by distance_bucket
    CREATE OR REPLACE VIEW metrics_by_distance AS
    SELECT 
        'distance_bucket' AS segment_type,
        distance_bucket AS segment_value,
        COUNT(*) AS trip_count,
        AVG(abs_error_sec) AS mae_sec,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY abs_error_sec) AS medae_sec,
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY abs_error_sec) AS p90_abs_error_sec,
        AVG(abs_pct_error) AS mae_pct,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY abs_pct_error) AS medae_pct,
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY abs_pct_error) AS p90_abs_error_pct,
        AVG(CASE WHEN abs_pct_error <= 0.10 THEN 1.0 ELSE 0.0 END) AS pct_within_10pct,
        AVG(CASE WHEN abs_pct_error <= 0.20 THEN 1.0 ELSE 0.0 END) AS pct_within_20pct
    FROM eta_errors
    GROUP BY distance_bucket
    ORDER BY 
        CASE distance_bucket
            WHEN '<1mi' THEN 1
            WHEN '1-3mi' THEN 2
            WHEN '3-5mi' THEN 3
            WHEN '5-10mi' THEN 4
            WHEN '10+mi' THEN 5
        END;
    
    -- Aggregate metrics by hour_of_day
    CREATE OR REPLACE VIEW metrics_by_hour AS
    SELECT 
        'hour_of_day' AS segment_type,
        hour_of_day::TEXT AS segment_value,
        COUNT(*) AS trip_count,
        AVG(abs_error_sec) AS mae_sec,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY abs_error_sec) AS medae_sec,
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY abs_error_sec) AS p90_abs_error_sec,
        AVG(abs_pct_error) AS mae_pct,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY abs_pct_error) AS medae_pct,
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY abs_pct_error) AS p90_abs_error_pct,
        AVG(CASE WHEN abs_pct_error <= 0.10 THEN 1.0 ELSE 0.0 END) AS pct_within_10pct,
        AVG(CASE WHEN abs_pct_error <= 0.20 THEN 1.0 ELSE 0.0 END) AS pct_within_20pct
    FROM eta_errors
    GROUP BY hour_of_day
    ORDER BY hour_of_day;
    
    -- Aggregate metrics by day_of_week
    CREATE OR REPLACE VIEW metrics_by_day_of_week AS
    SELECT 
        'day_of_week' AS segment_type,
        CASE day_of_week
            WHEN 0 THEN 'Monday'
            WHEN 1 THEN 'Tuesday'
            WHEN 2 THEN 'Wednesday'
            WHEN 3 THEN 'Thursday'
            WHEN 4 THEN 'Friday'
            WHEN 5 THEN 'Saturday'
            WHEN 6 THEN 'Sunday'
        END AS segment_value,
        COUNT(*) AS trip_count,
        AVG(abs_error_sec) AS mae_sec,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY abs_error_sec) AS medae_sec,
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY abs_error_sec) AS p90_abs_error_sec,
        AVG(abs_pct_error) AS mae_pct,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY abs_pct_error) AS medae_pct,
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY abs_pct_error) AS p90_abs_error_pct,
        AVG(CASE WHEN abs_pct_error <= 0.10 THEN 1.0 ELSE 0.0 END) AS pct_within_10pct,
        AVG(CASE WHEN abs_pct_error <= 0.20 THEN 1.0 ELSE 0.0 END) AS pct_within_20pct
    FROM eta_errors
    GROUP BY day_of_week
    ORDER BY day_of_week;
    
    -- Combined metrics view for easy export
    CREATE OR REPLACE VIEW metrics_combined AS
    SELECT * FROM metrics_overall
    UNION ALL
    SELECT * FROM metrics_by_distance
    UNION ALL
    SELECT * FROM metrics_by_hour
    UNION ALL
    SELECT * FROM metrics_by_day_of_week;
    """
    
    # Execute SQL using SQLAlchemy 2.0 syntax
    with engine.begin() as conn:
        try:
            conn.execute(text(sql))
        except Exception as e:
            error_msg = str(e).lower()
            if 'already exists' not in error_msg and 'does not exist' not in error_msg:
                print(f"Warning executing SQL: {e}")
                if 'syntax error' in error_msg or 'relation' in error_msg:
                    raise


def load_metrics_from_db(engine):
    """
    Load all metrics from database views.
    
    Args:
        engine: SQLAlchemy engine
    
    Returns:
        Dictionary of DataFrames with metrics
    """
    metrics = {}
    
    views = [
        'metrics_overall',
        'metrics_by_distance',
        'metrics_by_hour',
        'metrics_by_day_of_week',
        'metrics_combined'
    ]
    
    with engine.connect() as conn:
        for view_name in views:
            try:
                query = f"SELECT * FROM {view_name}"
                df = pd.read_sql(query, conn)
                metrics[view_name] = df
                print(f"Loaded {len(df)} rows from {view_name}")
            except Exception as e:
                print(f"Warning: Could not load {view_name}: {e}")
    
    return metrics


def save_metrics_to_csv(metrics, output_dir):
    """
    Save metrics DataFrames to CSV files.
    
    Args:
        metrics: Dictionary of DataFrames
        output_dir: Directory to save CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for view_name, df in metrics.items():
        output_path = os.path.join(output_dir, f"{view_name}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {output_path}")


def print_metrics_summary(metrics):
    """
    Print a summary of key metrics.
    
    Args:
        metrics: Dictionary of DataFrames with metrics
    """
    if 'metrics_overall' in metrics:
        overall = metrics['metrics_overall']
        if len(overall) > 0:
            print("\n" + "="*60)
            print("OVERALL ETA ACCURACY METRICS")
            print("="*60)
            row = overall.iloc[0]
            print(f"Total evaluation trips:     {int(row['trip_count']):,}")
            print(f"Mean Absolute Error:        {row['mae_sec']:.1f} seconds ({row['mae_pct']:.1%})")
            print(f"Median Absolute Error:      {row['medae_sec']:.1f} seconds ({row['medae_pct']:.1%})")
            print(f"P90 Absolute Error:         {row['p90_abs_error_sec']:.1f} seconds ({row['p90_abs_error_pct']:.1%})")
            print(f"% within ±10%:              {row['pct_within_10pct']:.1%}")
            print(f"% within ±20%:              {row['pct_within_20pct']:.1%}")
            print("="*60)
    
    if 'metrics_by_distance' in metrics:
        print("\n" + "="*60)
        print("METRICS BY DISTANCE BUCKET")
        print("="*60)
        df = metrics['metrics_by_distance']
        for _, row in df.iterrows():
            print(f"\n{row['segment_value']}:")
            print(f"  Trips: {int(row['trip_count']):,}")
            print(f"  MedAE: {row['medae_sec']:.1f}s ({row['medae_pct']:.1%})")
            print(f"  P90:   {row['p90_abs_error_sec']:.1f}s ({row['p90_abs_error_pct']:.1%})")
            print(f"  ±10%:  {row['pct_within_10pct']:.1%}")
        print("="*60)


def main():
    """
    Main function to compute and save metrics.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute ETA accuracy metrics using ML model')
    parser.add_argument(
        '--model-path',
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'models', 'eta_model'),
        help='Path to trained model (without .pth extension)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=METRICS_DIR,
        help='Directory to save metrics CSV files'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='Batch size for predictions'
    )
    parser.add_argument(
        '--use-spark',
        action='store_true',
        help='Use Spark for loading data (faster for large datasets)'
    )
    parser.add_argument(
        '--spark-master',
        type=str,
        default='local[*]',
        help='Spark master URL (default: local[*])'
    )
    
    args = parser.parse_args()
    
    # Determine whether to use Spark
    use_spark = args.use_spark and SPARK_AVAILABLE
    spark = None
    
    if args.use_spark and not SPARK_AVAILABLE:
        print("Warning: --use-spark specified but PySpark not available. Using standard SQL.")
        use_spark = False
    
    if use_spark:
        print("Initializing Spark session...")
        spark = SparkSession.builder \
            .appName("ETA Metrics Computation") \
            .master(args.spark_master) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.jars.packages", "org.postgresql:postgresql:42.7.1") \
            .getOrCreate()
    
    # Create database connection
    print("Connecting to Postgres database...")
    engine = create_engine(DB_CONNECTION_STRING)
    
    # Test connection
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM trips_evaluation"))
            count = result.scalar()
            print(f"Connected. Found {count:,} trips in evaluation set")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("Please ensure:")
        print("  1. Postgres is running")
        print("  2. Database 'eta_analysis' exists")
        print("  3. Connection credentials in config.py are correct")
        print("  4. You've run load_to_postgres.py and 01_create_tables.sql")
        sys.exit(1)
    
    # Load trained model
    print(f"\nLoading trained model from {args.model_path}...")
    try:
        model, scaler, feature_names = load_model(args.model_path)
        print(f"Model loaded successfully. Input features: {len(feature_names)}")
    except FileNotFoundError:
        print(f"Error: Model not found at {args.model_path}")
        print("Please train the model first by running: python src/train_model.py")
        sys.exit(1)
    
    # Load evaluation data
    print("\nLoading evaluation data...")
    try:
        eval_df = load_evaluation_data(engine, use_spark=use_spark, spark=spark)
    except Exception as e:
        if spark is not None:
            spark.stop()
        raise e
    
    # Make predictions in batches
    print("\nMaking predictions...")
    predictions = []
    batch_size = args.batch_size
    
    for i in range(0, len(eval_df), batch_size):
        batch_df = eval_df.iloc[i:i+batch_size]
        batch_predictions = predict(model, scaler, batch_df)
        predictions.extend(batch_predictions)
        
        if (i + batch_size) % 50000 == 0 or i + batch_size >= len(eval_df):
            print(f"  Processed {min(i + batch_size, len(eval_df)):,} / {len(eval_df):,} trips")
    
    eval_df['estimated_duration_sec'] = predictions
    print(f"Generated {len(predictions):,} predictions")
    
    # Store predictions in database (keep Spark alive for this)
    print("\nStoring predictions in database...")
    try:
        store_predictions(engine, eval_df, use_spark=use_spark, spark=spark)
    finally:
        # Stop Spark after storing predictions
        if spark is not None:
            spark.stop()
    
    # Create metrics views
    print("\nComputing metrics...")
    create_metrics_views(engine)
    
    # Load metrics from database
    print("\nLoading metrics from database...")
    metrics = load_metrics_from_db(engine)
    
    # Print summary
    print_metrics_summary(metrics)
    
    # Save to CSV
    print(f"\nSaving metrics to {args.output_dir}...")
    save_metrics_to_csv(metrics, args.output_dir)
    
    print("\nMetrics computation complete!")


if __name__ == '__main__':
    main()
