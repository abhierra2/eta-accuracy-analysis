"""
Generate analysis plots and insights from ETA accuracy metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DB_CONNECTION_STRING, FIGURES_DIR, METRICS_DIR


def load_error_data(engine):
    """
    Load error data from database for plotting.
    
    Args:
        engine: SQLAlchemy engine
    
    Returns:
        DataFrame with error data
    """
    query = """
    SELECT 
        abs_error_sec,
        abs_pct_error,
        distance_bucket,
        hour_of_day,
        day_of_week,
        trip_duration_sec,
        estimated_duration_sec,
        haversine_distance_km
    FROM eta_errors
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    return df


def plot_error_histogram(df, output_dir):
    """
    Plot histogram of absolute ETA error.
    
    Args:
        df: DataFrame with error data
        output_dir: Directory to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absolute error in seconds
    axes[0].hist(df['abs_error_sec'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Absolute Error (seconds)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Absolute ETA Error (seconds)')
    axes[0].axvline(df['abs_error_sec'].median(), color='red', linestyle='--', 
                    label=f'Median: {df["abs_error_sec"].median():.1f}s')
    axes[0].axvline(df['abs_error_sec'].quantile(0.9), color='orange', linestyle='--',
                    label=f'P90: {df["abs_error_sec"].quantile(0.9):.1f}s')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Absolute error as percentage
    axes[1].hist(df['abs_pct_error'] * 100, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Absolute Error (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Absolute ETA Error (%)')
    axes[1].axvline(df['abs_pct_error'].median() * 100, color='red', linestyle='--',
                    label=f'Median: {df["abs_pct_error"].median()*100:.1f}%')
    axes[1].axvline(df['abs_pct_error'].quantile(0.9) * 100, color='orange', linestyle='--',
                    label=f'P90: {df["abs_pct_error"].quantile(0.9)*100:.1f}%')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'error_histogram.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_error_by_distance(df, output_dir):
    """
    Plot error metrics by distance bucket.
    
    Args:
        df: DataFrame with error data
        output_dir: Directory to save figure
    """
    # Aggregate by distance bucket
    distance_agg = df.groupby('distance_bucket').agg({
        'abs_error_sec': ['median', lambda x: x.quantile(0.9)],
        'abs_pct_error': ['median', lambda x: x.quantile(0.9)],
        'trip_duration_sec': 'count'
    }).reset_index()
    
    distance_agg.columns = ['distance_bucket', 'medae_sec', 'p90_sec', 'medae_pct', 'p90_pct', 'trip_count']
    
    # Order buckets
    bucket_order = ['<1mi', '1-3mi', '3-5mi', '5-10mi', '10+mi']
    distance_agg['order'] = distance_agg['distance_bucket'].map(
        {b: i for i, b in enumerate(bucket_order)}
    )
    distance_agg = distance_agg.sort_values('order')
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # MedAE and P90 in seconds
    x = np.arange(len(distance_agg))
    width = 0.35
    axes[0].bar(x - width/2, distance_agg['medae_sec'], width, label='MedAE', alpha=0.8)
    axes[0].bar(x + width/2, distance_agg['p90_sec'], width, label='P90', alpha=0.8)
    axes[0].set_xlabel('Distance Bucket')
    axes[0].set_ylabel('Error (seconds)')
    axes[0].set_title('ETA Error by Distance Bucket (seconds)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(distance_agg['distance_bucket'])
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis='y')
    
    # MedAE and P90 as percentage
    axes[1].bar(x - width/2, distance_agg['medae_pct'] * 100, width, label='MedAE', alpha=0.8)
    axes[1].bar(x + width/2, distance_agg['p90_pct'] * 100, width, label='P90', alpha=0.8)
    axes[1].set_xlabel('Distance Bucket')
    axes[1].set_ylabel('Error (%)')
    axes[1].set_title('ETA Error by Distance Bucket (%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(distance_agg['distance_bucket'])
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'error_by_distance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_error_by_hour(df, output_dir):
    """
    Plot error metrics by hour of day.
    
    Args:
        df: DataFrame with error data
        output_dir: Directory to save figure
    """
    # Aggregate by hour
    hour_agg = df.groupby('hour_of_day').agg({
        'abs_error_sec': ['median', lambda x: x.quantile(0.9)],
        'abs_pct_error': ['median', lambda x: x.quantile(0.9)],
        'trip_duration_sec': 'count'
    }).reset_index()
    
    hour_agg.columns = ['hour_of_day', 'medae_sec', 'p90_sec', 'medae_pct', 'p90_pct', 'trip_count']
    hour_agg = hour_agg.sort_values('hour_of_day')
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # MedAE and P90 in seconds
    axes[0].plot(hour_agg['hour_of_day'], hour_agg['medae_sec'], 
                 marker='o', label='MedAE', linewidth=2, markersize=6)
    axes[0].plot(hour_agg['hour_of_day'], hour_agg['p90_sec'], 
                 marker='s', label='P90', linewidth=2, markersize=6)
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Error (seconds)')
    axes[0].set_title('ETA Error by Hour of Day (seconds)')
    axes[0].set_xticks(range(0, 24))
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # MedAE and P90 as percentage
    axes[1].plot(hour_agg['hour_of_day'], hour_agg['medae_pct'] * 100, 
                 marker='o', label='MedAE', linewidth=2, markersize=6)
    axes[1].plot(hour_agg['hour_of_day'], hour_agg['p90_pct'] * 100, 
                 marker='s', label='P90', linewidth=2, markersize=6)
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Error (%)')
    axes[1].set_title('ETA Error by Hour of Day (%)')
    axes[1].set_xticks(range(0, 24))
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'error_by_hour.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_calibration(df, output_dir):
    """
    Plot calibration plot: predicted vs actual duration.
    
    Args:
        df: DataFrame with error data
        output_dir: Directory to save figure
    """
    # Sample for plotting if too many points
    if len(df) > 10000:
        df_plot = df.sample(n=10000, random_state=42)
    else:
        df_plot = df.copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    axes[0].scatter(df_plot['trip_duration_sec'], df_plot['estimated_duration_sec'],
                    alpha=0.3, s=1)
    max_duration = max(df_plot['trip_duration_sec'].max(), 
                      df_plot['estimated_duration_sec'].max())
    axes[0].plot([0, max_duration], [0, max_duration], 
                 'r--', linewidth=2, label='Perfect prediction')
    axes[0].set_xlabel('Actual Duration (seconds)')
    axes[0].set_ylabel('Estimated Duration (seconds)')
    axes[0].set_title('Predicted vs Actual Duration (scatter)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Binned calibration plot
    bins = np.linspace(0, df_plot['trip_duration_sec'].quantile(0.95), 20)
    df_plot['duration_bin'] = pd.cut(df_plot['trip_duration_sec'], bins=bins)
    bin_agg = df_plot.groupby('duration_bin').agg({
        'trip_duration_sec': 'mean',
        'estimated_duration_sec': 'mean'
    }).reset_index()
    
    axes[1].scatter(bin_agg['trip_duration_sec'], bin_agg['estimated_duration_sec'],
                    s=100, alpha=0.7, edgecolors='black')
    max_duration = max(bin_agg['trip_duration_sec'].max(),
                      bin_agg['estimated_duration_sec'].max())
    axes[1].plot([0, max_duration], [0, max_duration],
                 'r--', linewidth=2, label='Perfect prediction')
    axes[1].set_xlabel('Actual Duration (seconds)')
    axes[1].set_ylabel('Estimated Duration (seconds)')
    axes[1].set_title('Predicted vs Actual Duration (binned)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'calibration_plot.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_insights(df, metrics_dir):
    """
    Print interpretable insights from the analysis.
    
    Args:
        df: DataFrame with error data
        metrics_dir: Directory containing metrics CSV files
    """
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    # Overall statistics
    print(f"\n1. Overall Performance:")
    print(f"   - Median absolute error: {df['abs_error_sec'].median():.1f} seconds "
          f"({df['abs_pct_error'].median()*100:.1f}%)")
    print(f"   - P90 absolute error: {df['abs_error_sec'].quantile(0.9):.1f} seconds "
          f"({df['abs_pct_error'].quantile(0.9)*100:.1f}%)")
    print(f"   - {df[df['abs_pct_error'] <= 0.10].shape[0] / len(df) * 100:.1f}% of trips "
          f"within ±10% of actual duration")
    
    # Distance bucket insights
    print(f"\n2. Performance by Distance:")
    distance_medae = df.groupby('distance_bucket')['abs_pct_error'].median().sort_values()
    worst_distance = distance_medae.index[-1]
    best_distance = distance_medae.index[0]
    print(f"   - Best performing: {best_distance} "
          f"(MedAE: {distance_medae[best_distance]*100:.1f}%)")
    print(f"   - Worst performing: {worst_distance} "
          f"(MedAE: {distance_medae[worst_distance]*100:.1f}%)")
    
    # Hour of day insights
    print(f"\n3. Performance by Time of Day:")
    hour_medae = df.groupby('hour_of_day')['abs_pct_error'].median()
    worst_hour = hour_medae.idxmax()
    best_hour = hour_medae.idxmin()
    print(f"   - Best hour: {best_hour}:00 "
          f"(MedAE: {hour_medae[best_hour]*100:.1f}%)")
    print(f"   - Worst hour: {worst_hour}:00 "
          f"(MedAE: {hour_medae[worst_hour]*100:.1f}%)")
    
    # Bias analysis
    df['error_sec'] = df['estimated_duration_sec'] - df['trip_duration_sec']
    mean_error = df['error_sec'].mean()
    if abs(mean_error) > 5:
        direction = "overestimate" if mean_error > 0 else "underestimate"
        print(f"\n4. Bias Detection:")
        print(f"   - Model tends to {direction} trip duration by "
              f"{abs(mean_error):.1f} seconds on average")
    
    print("="*60)


def main():
    """
    Main function to generate analysis plots and insights.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate ETA accuracy analysis plots')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=FIGURES_DIR,
        help='Directory to save figures'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create database connection
    print("Connecting to Postgres database...")
    engine = create_engine(DB_CONNECTION_STRING)
    
    # Test connection
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM eta_errors"))
            count = result.scalar()
            print(f"Connected. Found {count:,} trips with ETA predictions")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("Please run compute_metrics.py first to generate ETA predictions")
        sys.exit(1)
    
    # Load error data
    print("\nLoading error data from database...")
    df = load_error_data(engine)
    print(f"Loaded {len(df):,} trips")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_error_histogram(df, args.output_dir)
    plot_error_by_distance(df, args.output_dir)
    plot_error_by_hour(df, args.output_dir)
    plot_calibration(df, args.output_dir)
    
    # Print insights
    print_insights(df, METRICS_DIR)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()

