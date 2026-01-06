# ETA Accuracy & Route Quality Analysis

A lightweight end-to-end analytics project for evaluating ETA accuracy and route-quality proxy metrics using NYC Taxi trip data. This project is designed to demonstrate analytical capabilities relevant to a Mapping Data Scientist role.

## Problem Framing

In rideshare and mapping applications, accurate Estimated Time of Arrival (ETA) predictions are critical for:
- **User Experience**: Setting accurate expectations for trip duration
- **Driver Experience**: Enabling better route planning and earnings optimization
- **Business Operations**: Improving dispatch efficiency and reducing cancellations

This project evaluates ETA accuracy using historical trip data by:
1. Training a PyTorch neural network model to predict trip duration
2. Computing accuracy metrics across different trip segments
3. Identifying failure modes and areas for improvement
4. Proposing an experiment design for production deployment

## Dataset Description

This project uses NYC Taxi trip data, which typically contains:
- **Temporal features**: Pickup and dropoff timestamps
- **Geographic features**: Pickup and dropoff coordinates (latitude/longitude)
- **Trip characteristics**: Trip distance (may be in miles or kilometers)

The data is expected to be placed at `data/raw.csv`. The project handles common variations in column naming conventions (e.g., `tpep_pickup_datetime`, `lpep_pickup_datetime`, etc.) through automatic column mapping.

### Expected Data Format

The script will automatically map common column name variations. Standard columns expected:
- `pickup_datetime` / `tpep_pickup_datetime` / `lpep_pickup_datetime`
- `dropoff_datetime` / `tpep_dropoff_datetime` / `lpep_dropoff_datetime`
- `pickup_latitude` / `start_lat`
- `pickup_longitude` / `start_lon`
- `dropoff_latitude` / `end_lat`
- `dropoff_longitude` / `end_lon`
- `trip_distance` / `distance`

## Project Structure

```
mapping_project/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── raw.csv              # Place your NYC taxi dataset here
├── sql/
│   ├── 01_create_tables.sql # Schema setup and derived columns
│   └── 02_metrics.sql       # ETA accuracy metrics computation
├── src/
│   ├── config.py            # Configuration and column mappings
│   ├── load_to_postgres.py  # Data loading and cleaning
│   ├── model.py             # PyTorch neural network model
│   ├── train_model.py       # Model training script
│   ├── compute_metrics.py   # Metrics computation with ML model
│   ├── analysis.py          # Visualization and insights
│   └── utils_geo.py         # Geographic utilities (Haversine)
└── outputs/
    ├── figures/             # Generated plots
    ├── metrics/             # Computed metrics (CSV)
    └── models/              # Trained model files
```

## Setup Instructions

### Prerequisites

1. **Python 3.10+** installed
2. **PostgreSQL** installed and running locally
3. **pgAdmin** (optional, for database management)
4. A PostgreSQL database named `eta_analysis` created
5. **Java 8+** (required for Spark, if using `--use-spark` flag)

### Installation

1. **Clone or navigate to the project directory**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up PostgreSQL database**
   ```sql
   CREATE DATABASE eta_analysis;
   ```

5. **Configure database connection** (if needed)
   
   Edit `src/config.py` to match your PostgreSQL credentials:
   ```python
   DB_CONFIG = {
       'host': 'localhost',
       'port': 5432,
       'database': 'eta_analysis',
       'user': 'postgres',  # Update if different
       'password': 'postgres'  # Update if different
   }
   ```
   
   Or set environment variables:
   ```bash
   export POSTGRES_USER=your_username
   export POSTGRES_PASSWORD=your_password
   ```

6. **Download and place NYC taxi data**
   
   Download a NYC taxi trip dataset (e.g., from [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)) and place it at `data/raw.csv`

### Spark Setup (Optional, for Large Datasets)

Spark is **optional** but recommended for processing large datasets (>1M rows). The project will automatically fall back to Pandas if Spark is not available.

1. **Install Java 8+** (required for Spark)
   ```bash
   # macOS
   brew install openjdk@11
   
   # Linux (Ubuntu/Debian)
   sudo apt-get install openjdk-11-jdk
   ```

2. **PySpark is included in requirements.txt** and will be installed with:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Java home** (if needed):
   ```bash
   export JAVA_HOME=/path/to/java
   ```

**Note**: Spark is automatically used when you specify the `--use-spark` flag. Without this flag, the scripts use Pandas, which is sufficient for smaller datasets.

## How to Run

### Step 1: Load Data to Postgres

```bash
python src/load_to_postgres.py
```

Or with Spark for faster processing of large datasets:

```bash
python src/load_to_postgres.py --use-spark
```

This script will:
- Read `data/raw.csv`
- Normalize column names
- Compute trip duration and Haversine distance
- Filter invalid rows (negative duration, missing coordinates, unrealistic trips)
- Load cleaned data into `trips_clean` table

**Optional flags:**
- `--input PATH`: Specify custom input file path
- `--table NAME`: Specify custom table name
- `--sample N`: Sample N rows for testing
- `--use-spark`: Use Apache Spark for processing (much faster for large datasets)
- `--spark-master URL`: Spark master URL (default: `local[*]`)

### Step 2: Create Tables and Derived Columns

```bash
psql -U postgres -d eta_analysis -f sql/01_create_tables.sql
```

Or execute via Python:
```python
from sqlalchemy import create_engine, text
from src.config import DB_CONNECTION_STRING

engine = create_engine(DB_CONNECTION_STRING)
with open('sql/01_create_tables.sql', 'r') as f:
    with engine.connect() as conn:
        conn.execute(text(f.read()))
        conn.commit()
```

This creates:
- Derived columns (hour_of_day, day_of_week, distance_bucket)
- Indexes for performance
- Training/evaluation data splits (70/30)

### Step 3: Train the ML Model

```bash
python src/train_model.py
```

This script will:
- Load training data from the database
- Prepare features (distance, time, cyclical encodings, etc.)
- Train a PyTorch neural network model
- Save the trained model to `outputs/models/eta_model.pth`

**Optional flags:**
- `--batch-size N`: Batch size for training (default: 256)
- `--learning-rate LR`: Learning rate (default: 0.001)
- `--epochs N`: Number of training epochs (default: 50)
- `--val-split RATIO`: Validation split ratio (default: 0.2)

### Step 4: Compute Metrics

```bash
python src/compute_metrics.py
```

Or with Spark for faster data loading:

```bash
python src/compute_metrics.py --use-spark
```

This script will:
- Load the trained PyTorch model
- Generate ETA predictions for evaluation trips
- Calculate accuracy metrics (MAE, MedAE, P90, % within thresholds)
- Segment metrics by distance, hour, and day of week
- Save metrics to `outputs/metrics/*.csv`

**Optional flags:**
- `--model-path PATH`: Path to trained model (default: outputs/models/eta_model)
- `--batch-size N`: Batch size for predictions (default: 10000)
- `--use-spark`: Use Spark for loading data from database (faster for large datasets)
- `--spark-master URL`: Spark master URL (default: `local[*]`)

### Step 5: Generate Analysis and Visualizations

```bash
python src/analysis.py
```

This script will:
- Generate plots:
  - Histogram of absolute ETA error
  - Error by distance bucket
  - Error by hour of day
  - Calibration plot (predicted vs actual)
- Print key insights
- Save figures to `outputs/figures/`

## Model Architecture

The project uses a **PyTorch neural network** for ETA prediction:

### Architecture
- **Type**: Feedforward neural network (multi-layer perceptron)
- **Input Features**:
  - Haversine distance (km)
  - Hour of day (with cyclical sin/cos encoding)
  - Day of week (with cyclical sin/cos encoding)
  - Distance bucket (one-hot encoded: <1mi, 1-3mi, 3-5mi, 5-10mi, 10+mi)
  - Pickup/dropoff coordinates (normalized to NYC area)
- **Hidden Layers**: 3 layers with sizes [128, 64, 32]
- **Activation**: ReLU with dropout (0.2) for regularization
- **Output**: Single value (predicted trip duration in seconds)
- **Loss Function**: Mean Absolute Error (MAE / L1 loss)
- **Optimizer**: Adam with learning rate 0.001

### Feature Engineering
- **Cyclical encoding**: Time features (hour, day) are encoded using sin/cos to capture cyclical patterns
- **Standardization**: All features are standardized using StandardScaler
- **One-hot encoding**: Categorical distance buckets are one-hot encoded

### Training
- **Train/Val/Test Split**: 70% training, 20% validation (from training), 30% evaluation
- **Early Stopping**: Stops training if validation loss doesn't improve for 10 epochs
- **Batch Size**: 256 (configurable)
- **Epochs**: 50 (configurable)

## Metric Definitions

### Primary Metrics

1. **Mean Absolute Error (MAE)**
   - Average absolute difference between predicted and actual trip duration
   - Units: seconds or percentage
   - Interpretation: Overall prediction accuracy

2. **Median Absolute Error (MedAE)**
   - Median absolute difference (robust to outliers)
   - Units: seconds or percentage
   - Interpretation: Typical prediction error

3. **P90 Absolute Error**
   - 90th percentile of absolute errors
   - Units: seconds or percentage
   - Interpretation: Worst-case performance for 90% of trips

4. **% Within ±10% / ±20%**
   - Percentage of trips with error within 10% or 20% of actual duration
   - Interpretation: Coverage of acceptable predictions

### Segmentation Dimensions

Metrics are computed across:
- **Distance buckets**: <1mi, 1-3mi, 3-5mi, 5-10mi, 10+mi
- **Hour of day**: 0-23
- **Day of week**: Monday-Sunday

## Summary of Findings

*[This section will be populated after running the analysis]*

### Overall Performance
- Median absolute error: [X] seconds ([Y]%)
- P90 absolute error: [X] seconds ([Y]%)
- [Z]% of trips within ±10% of actual duration

### Key Patterns
- **Distance**: [Short/long trips show better/worse accuracy]
- **Time of day**: [Peak hours show higher/lower errors]
- **Day of week**: [Weekday/weekend differences]

### Failure Modes Identified

1. **Short trips (<1 mile)**
   - Higher relative error due to fixed overhead (traffic lights, stops)
   - Recommendation: Use different model or add minimum duration buffer

2. **Peak hours (rush hour)**
   - Traffic variability leads to higher errors
   - Recommendation: Time-of-day specific speed models

3. **Dense urban areas**
   - Complex routing and frequent stops
   - Recommendation: Incorporate traffic data and route complexity features

4. **Long trips (10+ miles)**
   - Highway vs. city street mix creates variability
   - Recommendation: Segment by road type or use route-based features

## Proposed Online Experiment

### Experiment Design

**Objective**: Improve ETA accuracy by deploying segment-aware ETA adjustments

**Hypothesis**: Improving the ML model with additional features and calibration will improve overall accuracy without degrading user experience.

### Control vs Treatment

- **Control**: Current PyTorch neural network model (baseline features)
- **Treatment**: Enhanced model with additional features
  - Add real-time traffic data
  - Include route-based features (road type, intersections)
  - Apply post-hoc calibration adjustments per segment
  - Example: If short trips are consistently underestimated, add learned bias corrections

### Primary Metrics

1. **Median Absolute Error (MedAE)**
   - Primary success metric
   - Expected improvement: 5-10% reduction

2. **P90 Absolute Error**
   - Secondary success metric
   - Expected improvement: 5-15% reduction

### Guardrail Metrics

1. **Driver Cancel Rate**
   - Monitor for increases (may indicate driver frustration with inaccurate ETAs)
   - Threshold: <5% increase

2. **Reroute Rate**
   - Monitor for increases (may indicate route quality issues)
   - Threshold: <10% increase

3. **User Satisfaction Scores**
   - Monitor for decreases
   - Threshold: No significant decrease

### Experiment Parameters

- **Duration**: 4 weeks
- **Traffic**: 50/50 split (randomized by trip)
- **Geographic scope**: NYC metro area
- **Sample size**: ~100K trips per variant (power analysis required)

### Analysis Plan

1. **Primary analysis**: Compare MedAE and P90 between control and treatment
2. **Segmentation analysis**: Evaluate improvements by distance, time, geography
3. **Guardrail analysis**: Ensure no degradation in user experience metrics
4. **Statistical testing**: Use appropriate tests (t-test, Mann-Whitney U) with multiple comparison correction

## Scaling to Production

### Data Infrastructure

1. **Real-time data pipeline**
   - Stream trip data from production systems
   - Compute features in real-time (current traffic, weather, events)
   - Store in time-series database (e.g., InfluxDB) or data warehouse

2. **Feature store**
   - Historical speed profiles by segment
   - Real-time traffic conditions
   - Route characteristics (road type, intersections, elevation)
   - Temporal features (time, day, holidays, events)

3. **Model serving**
   - Deploy ETA models via ML serving platform (e.g., MLflow, Seldon)
   - A/B test different model variants
   - Monitor model performance in real-time

### Model Improvements

1. **Feature engineering**
   - Route-based features (road type, number of turns, traffic lights)
   - Real-time traffic data (Google Maps, Waze, internal telemetry)
   - Weather conditions
   - Historical patterns (day of week, holidays, events)
   - Embeddings for pickup/dropoff locations

2. **Model architecture**
   - Current: Feedforward neural network with 3 hidden layers
   - Future: Gradient boosting (XGBoost, LightGBM) for non-linear patterns
   - Future: Deep learning (LSTM/Transformer) for sequential route data
   - Future: Ensemble methods combining multiple models
   - Future: Graph neural networks for spatial relationships

3. **Continuous learning**
   - Retrain models weekly/monthly with recent data
   - Online learning for rapid adaptation to traffic patterns
   - Monitor for data drift and model degradation

### Monitoring and Alerting

1. **Model performance monitoring**
   - Track MAE, MedAE, P90 in real-time dashboards
   - Alert on significant degradation (>10% increase in error)
   - Segment-level monitoring to catch localized issues

2. **Business metrics**
   - Driver cancel rate
   - User satisfaction scores
   - Reroute requests
   - On-time arrival rate

3. **Data quality**
   - Monitor for missing data, outliers, schema changes
   - Validate feature distributions

### Experimentation Platform

1. **A/B testing infrastructure**
   - Feature flags for model variants
   - Randomized assignment by trip
   - Statistical analysis tools (CausalML, Eppo)

2. **Rapid iteration**
   - Deploy new models to small traffic (1-5%)
   - Gradual rollout with monitoring
   - Quick rollback capability

## Limitations and Future Work

### Current Limitations

1. **Neural network model**: Uses PyTorch feedforward network with basic features (distance, time, location)
2. **No route information**: Doesn't consider actual route taken or road characteristics
3. **No traffic data**: Doesn't incorporate current traffic conditions
4. **Static model**: Doesn't adapt to changing patterns over time (requires retraining)
5. **Limited features**: Only uses basic trip characteristics, not real-time context

### Future Enhancements

1. **Route-based features**: Incorporate actual route geometry and road types
2. **Real-time traffic**: Integrate traffic APIs (Google Maps, Waze)
3. **Machine learning models**: Replace median-based approach with ML models
4. **Uncertainty quantification**: Provide confidence intervals for ETAs
5. **Multi-modal routing**: Consider alternative routes and their probabilities

## Troubleshooting

### Common Issues

1. **Database connection errors**
   - Verify PostgreSQL is running: `pg_isready`
   - Check credentials in `src/config.py`
   - Ensure database `eta_analysis` exists

2. **Column mapping errors**
   - Check that your CSV has expected columns
   - Review `src/config.py` to add custom column mappings
   - Verify column names match one of the variations listed

3. **Memory errors with large datasets**
   - Use `--sample N` flag to test with smaller dataset
   - Process data in chunks (modify `load_to_postgres.py`)

4. **SQL execution errors**
   - Ensure `trips_clean` table exists (run Step 1 first)
   - Check PostgreSQL version (requires 9.5+ for PERCENTILE_CONT)

## License

This project is provided as-is for educational and demonstration purposes.

## Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.

