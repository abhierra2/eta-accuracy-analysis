# Quick Start Guide

## Prerequisites Checklist

- [ ] Python 3.10+ installed
- [ ] PostgreSQL installed and running
- [ ] Database `eta_analysis` created
- [ ] NYC taxi dataset downloaded and placed at `data/raw.csv`

## Setup (One-time)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure database (if needed)
# Edit src/config.py or set environment variables:
export POSTGRES_USER=your_username
export POSTGRES_PASSWORD=your_password
```

## Run Analysis (Step-by-step)

```bash
# Step 1: Load data to Postgres
# For large datasets, use --use-spark for faster processing:
python src/load_to_postgres.py --use-spark
# Or without Spark (works fine for smaller datasets):
python src/load_to_postgres.py

# Step 2: Create tables and derived columns
psql -U postgres -d eta_analysis -f sql/01_create_tables.sql
# OR execute via Python (see README for details)

# Step 3: Train the PyTorch model
python src/train_model.py

# Step 4: Compute metrics using trained model
# For large datasets, use --use-spark for faster data loading:
python src/compute_metrics.py --use-spark
# Or without Spark:
python src/compute_metrics.py

# Step 5: Generate analysis and plots
python src/analysis.py
```

## Expected Outputs

After running all steps, you should have:

- **Database tables**: `trips_clean`, views for training/evaluation data
- **Metrics CSV files**: `outputs/metrics/*.csv`
- **Plots**: `outputs/figures/*.png`
  - `error_histogram.png`
  - `error_by_distance.png`
  - `error_by_hour.png`
  - `calibration_plot.png`

## Troubleshooting

**Database connection error?**
- Check PostgreSQL is running: `pg_isready`
- Verify database exists: `psql -l | grep eta_analysis`
- Check credentials in `src/config.py`

**Column mapping error?**
- Check your CSV column names
- Add custom mappings to `src/config.py` if needed

**Memory error with large dataset?**
- Use `--sample 10000` flag: `python src/load_to_postgres.py --sample 10000`

