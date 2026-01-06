"""
Load and clean NYC taxi trip data, then load into Postgres database.
Supports both Spark and Pandas backends for processing.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DB_CONNECTION_STRING, COLUMN_MAPPING, DATA_QUALITY
from utils_geo import haversine_distance_km

# Try to import Spark
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, when, isnan, isnull, udf
    from pyspark.sql.types import DoubleType
    import math
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("Note: PySpark not available. Using Pandas backend.")


def normalize_column_names(df, is_spark=False):
    """
    Normalize column names to standard format using mapping from config.
    
    Args:
        df: DataFrame (Spark or Pandas) with potentially non-standard column names
        is_spark: Whether df is a Spark DataFrame
    
    Returns:
        DataFrame with normalized column names
    """
    # Create reverse mapping: original_name -> standard_name
    column_map = {}
    for standard_name, variations in COLUMN_MAPPING.items():
        for variation in variations:
            if is_spark:
                if variation in df.columns:
                    column_map[variation] = standard_name
                    break
            else:
                if variation in df.columns:
                    column_map[variation] = standard_name
                    break
    
    # Rename columns
    if is_spark:
        for old_name, new_name in column_map.items():
            df = df.withColumnRenamed(old_name, new_name)
    else:
        df = df.rename(columns=column_map)
    
    # Check if all required columns are present
    required_cols = list(COLUMN_MAPPING.keys())
    if is_spark:
        available_cols = df.columns
    else:
        available_cols = df.columns.tolist()
    
    missing_cols = [col for col in required_cols if col not in available_cols]
    
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {available_cols}"
        )
    
    return df


def process_with_spark(spark, input_path, sample_size=None):
    """
    Process data using Spark for better performance on large datasets.
    
    Args:
        spark: SparkSession
        input_path: Path to input CSV file
        sample_size: Optional sample size for testing
    
    Returns:
        Cleaned Spark DataFrame
    """
    print("Using Spark for data processing...")
    
    # Read CSV with Spark
    print(f"Reading CSV from {input_path}...")
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)
    
    if sample_size:
        df = df.limit(sample_size)
        print(f"Sampled {sample_size} rows for processing")
    
    initial_count = df.count()
    print(f"Loaded {initial_count:,} rows")
    print(f"Columns: {df.columns}")
    
    # Normalize column names
    print("\nNormalizing column names...")
    df = normalize_column_names(df, is_spark=True)
    print(f"Normalized columns: {df.columns}")
    
    # Register haversine UDF
    def haversine_udf_func(lat1, lon1, lat2, lon2):
        """Haversine distance calculation for Spark UDF."""
        R = 6371.0  # Earth's radius in km
        lat1_rad = math.radians(float(lat1) if lat1 else 0)
        lon1_rad = math.radians(float(lon1) if lon1 else 0)
        lat2_rad = math.radians(float(lat2) if lat2 else 0)
        lon2_rad = math.radians(float(lon2) if lon2 else 0)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    haversine_udf = udf(haversine_udf_func, DoubleType())
    
    # Parse datetime columns and compute derived columns
    print("\nParsing datetime columns and computing derived columns...")
    from pyspark.sql.functions import to_timestamp, unix_timestamp, when, col, isnan, isnull
    
    df = df.withColumn(
        "pickup_datetime",
        to_timestamp(col("pickup_datetime"), "yyyy-MM-dd HH:mm:ss")
    ).withColumn(
        "dropoff_datetime",
        to_timestamp(col("dropoff_datetime"), "yyyy-MM-dd HH:mm:ss")
    )
    
    # Compute trip duration
    df = df.withColumn(
        "trip_duration_sec",
        (unix_timestamp("dropoff_datetime") - unix_timestamp("pickup_datetime"))
    )
    
    # Compute haversine distance
    df = df.withColumn(
        "haversine_distance_km",
        haversine_udf(
            col("pickup_latitude"),
            col("pickup_longitude"),
            col("dropoff_latitude"),
            col("dropoff_longitude")
        )
    )
    
    # Filter invalid rows
    print("\nFiltering invalid rows...")
    stats = {'initial_count': initial_count}
    
    # Filter 1: Missing datetime values
    before_count = df.count()
    df = df.filter(
        col("pickup_datetime").isNotNull() & 
        col("dropoff_datetime").isNotNull()
    )
    stats['missing_datetime'] = before_count - df.count()
    
    # Filter 2: Invalid duration
    before_count = df.count()
    df = df.filter(
        (col("trip_duration_sec") > DATA_QUALITY['min_duration_sec']) &
        (col("trip_duration_sec") <= DATA_QUALITY['max_duration_sec'])
    )
    stats['invalid_duration'] = before_count - df.count()
    
    # Filter 3: Missing coordinates
    before_count = df.count()
    df = df.filter(
        col("pickup_latitude").isNotNull() &
        col("pickup_longitude").isNotNull() &
        col("dropoff_latitude").isNotNull() &
        col("dropoff_longitude").isNotNull()
    )
    stats['missing_coordinates'] = before_count - df.count()
    
    # Filter 4: Invalid coordinate ranges
    before_count = df.count()
    df = df.filter(
        (col("pickup_latitude").between(*DATA_QUALITY['valid_lat_range'])) &
        (col("pickup_longitude").between(*DATA_QUALITY['valid_lon_range'])) &
        (col("dropoff_latitude").between(*DATA_QUALITY['valid_lat_range'])) &
        (col("dropoff_longitude").between(*DATA_QUALITY['valid_lon_range']))
    )
    stats['invalid_coordinate_range'] = before_count - df.count()
    
    # Filter 5: Invalid distances
    before_count = df.count()
    df = df.filter(
        (col("haversine_distance_km").isNotNull()) &
        (col("haversine_distance_km") >= DATA_QUALITY['min_distance_km']) &
        (col("haversine_distance_km") <= DATA_QUALITY['max_distance_km'])
    )
    stats['invalid_distance'] = before_count - df.count()
    stats['nan_distance'] = 0  # Handled in invalid_distance filter
    
    final_count = df.count()
    stats['final_count'] = final_count
    stats['filtered_out'] = initial_count - final_count
    stats['retention_rate'] = final_count / initial_count if initial_count > 0 else 0
    
    # Generate trip_id as a sequential integer starting from 1
    print("\nGenerating trip_id...")
    from pyspark.sql.functions import row_number
    from pyspark.sql.window import Window
    
    # Order by pickup_datetime and coordinates to ensure consistent ordering
    df = df.withColumn(
        "trip_id",
        row_number().over(Window.orderBy("pickup_datetime", "pickup_latitude", "pickup_longitude"))
    )
    
    # Print filtering statistics
    print("\n" + "="*60)
    print("DATA FILTERING STATISTICS")
    print("="*60)
    print(f"Initial row count:      {stats['initial_count']:,}")
    print(f"Missing datetime:        {stats['missing_datetime']:,}")
    print(f"Invalid duration:        {stats['invalid_duration']:,}")
    print(f"Missing coordinates:     {stats['missing_coordinates']:,}")
    print(f"Invalid coordinate range: {stats['invalid_coordinate_range']:,}")
    print(f"Invalid distance:        {stats['invalid_distance']:,}")
    print(f"Final row count:         {stats['final_count']:,}")
    print(f"Filtered out:            {stats['filtered_out']:,}")
    print(f"Retention rate:          {stats['retention_rate']:.2%}")
    print("="*60)
    
    return df, stats


def process_with_pandas(input_path, sample_size=None):
    """
    Process data using Pandas (fallback for smaller datasets or when Spark unavailable).
    
    Args:
        input_path: Path to input CSV file
        sample_size: Optional sample size for testing
    
    Returns:
        Tuple of (cleaned DataFrame, filter_stats)
    """
    print("Using Pandas for data processing...")
    
    # Read CSV in chunks to handle large files
    try:
        chunk_size = 100000
        chunks = []
        for chunk in pd.read_csv(input_path, chunksize=chunk_size, low_memory=False):
            chunks.append(chunk)
            if sample_size and len(pd.concat(chunks, ignore_index=True)) >= sample_size:
                break
        
        df = pd.concat(chunks, ignore_index=True)
        
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
            print(f"Sampled {len(df)} rows for processing")
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Normalize column names
    print("\nNormalizing column names...")
    df = normalize_column_names(df, is_spark=False)
    print(f"Normalized columns: {list(df.columns)}")
    
    # Parse datetime columns
    print("\nParsing datetime columns...")
    for col_name in ['pickup_datetime', 'dropoff_datetime']:
        if col_name in df.columns:
            df[col_name] = pd.to_datetime(df[col_name], errors='coerce', infer_datetime_format=True)
    
    # Compute derived columns
    print("\nComputing derived columns (duration, haversine distance)...")
    df['trip_duration_sec'] = (
        (df['dropoff_datetime'] - df['pickup_datetime'])
        .dt.total_seconds()
    )
    
    df['haversine_distance_km'] = haversine_distance_km(
        df['pickup_latitude'],
        df['pickup_longitude'],
        df['dropoff_latitude'],
        df['dropoff_longitude']
    )
    
    # Filter invalid rows
    print("\nFiltering invalid rows...")
    initial_count = len(df)
    stats = {'initial_count': initial_count}
    
    # Filter 1: Missing datetime values
    before = len(df)
    df = df.dropna(subset=['pickup_datetime', 'dropoff_datetime'])
    stats['missing_datetime'] = before - len(df)
    
    # Filter 2: Negative or zero duration
    before = len(df)
    df = df[
        (df['trip_duration_sec'] > DATA_QUALITY['min_duration_sec']) &
        (df['trip_duration_sec'] <= DATA_QUALITY['max_duration_sec'])
    ]
    stats['invalid_duration'] = before - len(df)
    
    # Filter 3: Missing or invalid lat/lon
    before = len(df)
    df = df.dropna(subset=[
        'pickup_latitude', 'pickup_longitude',
        'dropoff_latitude', 'dropoff_longitude'
    ])
    stats['missing_coordinates'] = before - len(df)
    
    # Filter 4: Invalid coordinate ranges
    before = len(df)
    df = df[
        (df['pickup_latitude'].between(*DATA_QUALITY['valid_lat_range'])) &
        (df['pickup_longitude'].between(*DATA_QUALITY['valid_lon_range'])) &
        (df['dropoff_latitude'].between(*DATA_QUALITY['valid_lat_range'])) &
        (df['dropoff_longitude'].between(*DATA_QUALITY['valid_lon_range']))
    ]
    stats['invalid_coordinate_range'] = before - len(df)
    
    # Filter 5: Unrealistic distances
    before = len(df)
    df = df[
        (df['haversine_distance_km'] >= DATA_QUALITY['min_distance_km']) &
        (df['haversine_distance_km'] <= DATA_QUALITY['max_distance_km'])
    ]
    stats['invalid_distance'] = before - len(df)
    
    # Filter 6: Drop rows where haversine distance is NaN
    before = len(df)
    df = df.dropna(subset=['haversine_distance_km'])
    stats['nan_distance'] = before - len(df)
    
    final_count = len(df)
    stats['final_count'] = final_count
    stats['filtered_out'] = initial_count - final_count
    stats['retention_rate'] = final_count / initial_count if initial_count > 0 else 0
    
    # Generate trip_id as a sequential integer starting from 1
    print("\nGenerating trip_id...")
    # Reset index and generate trip_id
    df = df.reset_index(drop=True)
    df = df.sort_values(by=['pickup_datetime', 'pickup_latitude', 'pickup_longitude'])
    df = df.reset_index(drop=True)
    df['trip_id'] = df.index + 1
    
    # Print filtering statistics
    print("\n" + "="*60)
    print("DATA FILTERING STATISTICS")
    print("="*60)
    print(f"Initial row count:      {stats['initial_count']:,}")
    print(f"Missing datetime:        {stats['missing_datetime']:,}")
    print(f"Invalid duration:        {stats['invalid_duration']:,}")
    print(f"Missing coordinates:     {stats['missing_coordinates']:,}")
    print(f"Invalid coordinate range: {stats['invalid_coordinate_range']:,}")
    print(f"Invalid distance:        {stats['invalid_distance']:,}")
    print(f"NaN distance:            {stats['nan_distance']:,}")
    print(f"Final row count:         {stats['final_count']:,}")
    print(f"Filtered out:            {stats['filtered_out']:,}")
    print(f"Retention rate:          {stats['retention_rate']:.2%}")
    print("="*60)
    
    return df, stats


def load_to_postgres(df, table_name='trips_clean', if_exists='replace', is_spark=False, spark=None):
    """
    Load cleaned DataFrame to Postgres database.
    
    Args:
        df: Cleaned DataFrame (Spark or Pandas)
        table_name: Name of the table to create/overwrite
        if_exists: What to do if table exists ('replace', 'append', 'fail')
        is_spark: Whether df is a Spark DataFrame
        spark: SparkSession (required if is_spark=True)
    """
    engine = create_engine(DB_CONNECTION_STRING)
    
    if is_spark:
        # Try to use Spark JDBC connector first (more efficient for large datasets)
        row_count = df.count()
        print(f"\nLoading {row_count:,} rows to Postgres table '{table_name}'...")
        
        try:
            # Parse connection string
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
            
            # For overwrite mode, truncate table first (instead of dropping, which fails with dependent views)
            if if_exists == 'replace':
                try:
                    with engine.begin() as conn:
                        conn.execute(text(f"TRUNCATE TABLE {table_name}"))
                    print(f"Truncated existing table '{table_name}'")
                except Exception as truncate_error:
                    print(f"Note: Could not truncate table ({truncate_error}), will attempt overwrite")
            
            # Write using Spark JDBC (use append mode since we truncated if needed)
            df.write \
                .format("jdbc") \
                .option("url", jdbc_url) \
                .option("dbtable", table_name) \
                .option("user", properties["user"]) \
                .option("password", properties["password"]) \
                .option("driver", properties["driver"]) \
                .mode("append") \
                .save()
            
            print(f"Successfully loaded data to '{table_name}' using Spark JDBC")
        except Exception as e:
            print(f"Note: Spark JDBC write failed ({e})")
            print("Attempting batched write using foreachPartition...")
            
            if spark is None:
                raise ValueError("Spark session required for batched write fallback")
            
            # Parse connection details for partition writers
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
            
            # Get column names before partitioning (needed for serialization)
            column_names = df.columns
            
            # For overwrite mode, truncate table first
            if if_exists == 'replace':
                try:
                    with engine.begin() as conn:
                        conn.execute(text(f"TRUNCATE TABLE {table_name}"))
                    print(f"Truncated existing table '{table_name}'")
                except Exception as truncate_error:
                    print(f"Note: Could not truncate table ({truncate_error})")
            
            # Repartition into manageable chunks (100k rows per partition)
            batch_size = 100000
            num_partitions = max(1, min(200, (row_count // batch_size)))  # Cap at 200 partitions
            df_batched = df.coalesce(num_partitions)
            
            print(f"Writing {row_count:,} rows in {num_partitions} partitions...")
            
            # Create a write function that captures all needed variables
            # This avoids serialization issues with Spark objects
            def create_write_function(cols, tbl_name, conn_host, conn_port, conn_db, conn_user, conn_pass):
                """Create a partition write function with captured variables"""
                def write_partition(partition):
                    """Write a partition to Postgres"""
                    import pandas as pd
                    from sqlalchemy import create_engine
                    
                    rows = list(partition)
                    if not rows:
                        return
                    
                    # Create engine for this partition
                    partition_conn_str = f"postgresql://{conn_user}:{conn_pass}@{conn_host}:{conn_port}/{conn_db}"
                    partition_engine = create_engine(partition_conn_str)
                    
                    # Convert to Pandas using the captured column names
                    partition_df = pd.DataFrame(rows, columns=cols)
                    
                    # Write to Postgres (always append since we truncated if needed)
                    partition_df.to_sql(
                        tbl_name,
                        partition_engine,
                        if_exists='append',
                        index=False,
                        method='multi',
                        chunksize=10000
                    )
                return write_partition
            
            # Create the write function with all needed variables
            write_func = create_write_function(
                column_names, table_name, hostname, port, database, username, password
            )
            
            # Write all partitions in parallel (Spark handles this)
            df_batched.foreachPartition(write_func)
            
            print(f"Successfully loaded data to '{table_name}' using batched write")
    else:
        # Pandas DataFrame
        df_to_load = df.copy()
        print(f"\nLoading {len(df_to_load)} rows to Postgres table '{table_name}'...")
        
        df_to_load.to_sql(
            table_name,
            engine,
            if_exists=if_exists,
            index=False,
            method='multi',
            chunksize=10000
        )
        print(f"Successfully loaded data to '{table_name}'")
    
    # Verify load
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        count = result.scalar()
        print(f"Verified: {count:,} rows in database")


def main():
    """
    Main function to load and process NYC taxi data.
    """
    import argparse
    from config import DATA_DIR
    
    parser = argparse.ArgumentParser(description='Load NYC taxi data to Postgres')
    parser.add_argument(
        '--input',
        type=str,
        default=os.path.join(DATA_DIR, 'raw.csv'),
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--table',
        type=str,
        default='trips_clean',
        help='Postgres table name'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Sample N rows for testing (optional)'
    )
    parser.add_argument(
        '--use-spark',
        action='store_true',
        help='Use Spark for processing (faster for large datasets)'
    )
    parser.add_argument(
        '--spark-master',
        type=str,
        default='local[*]',
        help='Spark master URL (default: local[*])'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        print("Please place your NYC taxi dataset at data/raw.csv")
        sys.exit(1)
    
    # Determine whether to use Spark
    use_spark = args.use_spark and SPARK_AVAILABLE
    
    if args.use_spark and not SPARK_AVAILABLE:
        print("Warning: --use-spark specified but PySpark not available. Using Pandas.")
        use_spark = False
    
    # Process data
    if use_spark:
        # Initialize Spark session with PostgreSQL JDBC driver
        print("Initializing Spark session...")
        spark = SparkSession.builder \
            .appName("ETA Analysis Data Loading") \
            .master(args.spark_master) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.jars.packages", "org.postgresql:postgresql:42.7.1") \
            .getOrCreate()
        
        try:
            df, filter_stats = process_with_spark(spark, args.input, args.sample)
            
            # Load to Postgres
            print("\nLoading cleaned data to Postgres...")
            load_to_postgres(df, table_name=args.table, is_spark=True, spark=spark)
            
        finally:
            spark.stop()
    else:
        # Use Pandas
        df, filter_stats = process_with_pandas(args.input, args.sample)
        
        # Load to Postgres
        print("\nLoading cleaned data to Postgres...")
        load_to_postgres(df, table_name=args.table, is_spark=False)
    
    print("\nData loading complete!")


if __name__ == '__main__':
    main()
