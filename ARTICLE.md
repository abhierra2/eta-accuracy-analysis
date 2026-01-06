# Building an End-to-End ETA Accuracy Analysis System: Lessons from 3.5 Million NYC Taxi Trips

*How we built a production-ready analytics pipeline for evaluating ETA predictions using PyTorch, Spark, and PostgreSQL*

---

## The Problem: Why ETA Accuracy Matters

In rideshare and mapping applications, accurate Estimated Time of Arrival (ETA) predictions aren't just nice-to-have—they're critical for business success. Poor ETAs lead to frustrated users, driver cancellations, and lost revenue. But how do you systematically evaluate and improve ETA accuracy at scale?

This article chronicles our journey building a complete analytics system to evaluate ETA predictions using **3.5 million NYC taxi trips**. We'll walk through the technical challenges, architectural decisions, and key findings that emerged from the analysis.

## The Architecture: A Modern Data Science Stack

We built a lightweight but production-ready pipeline using:

- **PostgreSQL** for data storage and SQL-based analytics
- **PyTorch** for deep learning-based ETA prediction
- **Apache Spark** for distributed data processing
- **Python** for orchestration and analysis

The system follows a clean separation of concerns:

```
Data Loading → Feature Engineering → Model Training → Metrics Computation → Analysis
```

### Step 1: Data Loading and Cleaning

Our first challenge was handling **11.8 million raw trip records**. We implemented a dual-path approach:

- **Spark backend** for large-scale processing (10x faster for datasets >1M rows)
- **Pandas fallback** for smaller datasets or when Spark isn't available

Key data quality filters we applied:
- Removed trips with negative or zero duration
- Filtered out invalid coordinates (outside NYC bounds)
- Eliminated unrealistic trips (>2 hours, >100km)
- Computed Haversine distance for accurate geographic calculations

**Result**: 97.3% data retention rate (11.8M → 11.9M clean trips)

### Step 2: The Critical Discovery: Why `trip_id` Matters

Early in the project, we hit a puzzling issue: we stored 3.5 million predictions, but our metrics showed 22 million evaluation trips. What was going on?

The problem: we were joining predictions to trips using `pickup_datetime` as the key. But multiple trips can have the same pickup timestamp (imagine 10 taxis picking up at the same intersection at 5:00 PM). This created a **one-to-many join**, inflating our metrics.

**The Solution**: We generated a unique `trip_id` for each trip during data loading:

```python
# Spark version
df = df.withColumn(
    "trip_id",
    row_number().over(Window.orderBy("pickup_datetime", "pickup_latitude", "pickup_longitude"))
)

# Pandas version
df = df.sort_values(by=['pickup_datetime', 'pickup_latitude', 'pickup_longitude'])
df['trip_id'] = df.index + 1
```

This ensured one-to-one joins and accurate metric calculations. **Lesson learned**: Always use unique identifiers for joins, even when timestamps seem sufficient.

### Step 3: Building the Neural Network

We implemented a **feedforward neural network** using PyTorch:

**Architecture**:
- Input features: distance, time (cyclical encoding), location coordinates, distance buckets
- Hidden layers: [128, 64, 32] neurons with ReLU activation
- Dropout: 0.2 for regularization
- Output: Single value (predicted duration in seconds)

**Feature Engineering Highlights**:

1. **Cyclical Time Encoding**: Instead of treating hour (0-23) as a linear feature, we used sin/cos encoding:
   ```python
   hour_sin = sin(2π * hour / 24)
   hour_cos = cos(2π * hour / 24)
   ```
   This helps the model understand that 23:00 and 00:00 are close, not 23 hours apart.

2. **Distance Buckets**: Categorized trips into buckets (<1mi, 1-3mi, 3-5mi, 5-10mi, 10+mi) for better segmentation.

3. **Standardization**: All features standardized using `StandardScaler` for stable training.

**Training Enhancements**:

- **Learning Rate Scheduling**: `ReduceLROnPlateau` automatically reduces learning rate when validation loss plateaus
- **Early Stopping**: Saves best model and restores it if validation loss doesn't improve for 10 epochs
- **Best Model Checkpointing**: Automatically saves and restores the best-performing model weights

### Step 4: Spark Integration Challenges

Integrating Spark for performance brought several challenges:

**Challenge 1: Missing JDBC Driver**
```
java.lang.ClassNotFoundException: org.postgresql.Driver
```

**Solution**: Added PostgreSQL JDBC driver to Spark configuration:
```python
.config("spark.jars.packages", "org.postgresql:postgresql:42.7.1")
```

**Challenge 2: Out of Memory Errors**
When Spark JDBC failed, we tried converting the entire DataFrame to Pandas, causing OOM errors with 11.8M rows.

**Solution**: Implemented batched writes using `foreachPartition`:
```python
def create_write_function(cols, tbl_name, ...):
    def write_partition(partition):
        # Process one partition at a time
        partition_df = pd.DataFrame(rows, columns=cols)
        partition_df.to_sql(...)
    return write_partition
```

This processes data partition-by-partition, avoiding memory issues.

**Challenge 3: Table Dependencies**
Spark's `mode("overwrite")` tries to `DROP TABLE`, which fails when views depend on the table.

**Solution**: Manually truncate before writing:
```python
if if_exists == 'replace':
    conn.execute(text(f"TRUNCATE TABLE {table_name}"))
# Then use mode("append")
```

## Key Findings: What the Data Revealed

After processing 3.5 million evaluation trips, here's what we discovered:

### Overall Performance

- **Median Absolute Error**: 110 seconds (18% relative error)
- **P90 Absolute Error**: 381 seconds (51% relative error)
- **Coverage**: 29.4% of trips within ±10% of actual duration, 54.3% within ±20%

### Performance by Distance

The model performs **better on longer trips** (relative to trip duration):

| Distance Bucket | Median Error | % Within ±10% |
|----------------|--------------|---------------|
| <1mi | 24.4% | 22.2% |
| 1-3mi | 16.2% | 32.5% |
| 3-5mi | 14.4% | 35.9% |
| 5-10mi | 14.6% | 35.3% |
| 10+mi | 13.8% | 37.3% |

**Insight**: Short trips have higher relative error due to fixed overhead (traffic lights, stops) that doesn't scale with distance. A 30-second delay on a 2-minute trip is 25% error, but the same delay on a 20-minute trip is only 2.5%.

### Performance by Time of Day

Peak hours (8-9 AM, 4-5 PM) show **higher absolute errors** but similar relative errors:

- **Best hours**: 2-6 AM (MedAE: 13-16% relative error)
- **Worst hours**: 8-9 AM rush hour (MedAE: 19-20% relative error)
- **Evening hours**: 8-10 PM show good performance (MedAE: 16-17% relative error)

**Insight**: Traffic variability during rush hours creates prediction challenges. The model needs time-of-day specific calibration or real-time traffic features.

### Failure Modes Identified

1. **Short Trips (<1 mile)**: 24% median error, only 22% within ±10%
   - *Root cause*: Fixed overhead (traffic lights, stops) dominates trip time
   - *Recommendation*: Use different model or add minimum duration buffer

2. **Peak Hours**: Higher absolute errors during rush hour
   - *Root cause*: Traffic variability not captured by static features
   - *Recommendation*: Incorporate real-time traffic data

3. **Long Trips (10+ miles)**: Better relative error (13.8%) but high absolute error (332 seconds)
   - *Root cause*: Mix of highway and city streets creates variability
   - *Recommendation*: Segment by road type or use route-based features

## Technical Lessons Learned

### 1. Always Use Unique Identifiers for Joins

Using `pickup_datetime` as a join key seemed reasonable, but multiple trips can share the same timestamp. This caused metric inflation and incorrect analysis. **Always generate unique IDs** for proper one-to-one relationships.

### 2. Spark Requires Careful Memory Management

For large datasets, converting entire Spark DataFrames to Pandas causes OOM errors. Use `foreachPartition` for batched processing, or leverage Spark's native JDBC connector when possible.

### 3. Database Schema Dependencies Matter

When tables have dependent views, `DROP TABLE` fails. Use `TRUNCATE` instead to preserve schema while clearing data. This is especially important in production systems where views are used for analytics.

### 4. Feature Engineering is Critical

Cyclical encoding for time features dramatically improved model performance. Simple linear encoding would have treated 23:00 and 00:00 as 23 hours apart, missing the cyclical nature of daily patterns.

### 5. Model Training Needs Robustness

Early stopping and learning rate scheduling prevented overfitting and improved convergence. Always implement:
- Best model checkpointing
- Learning rate scheduling
- Validation monitoring

## Production Readiness: What's Next?

While this system works well for offline analysis, production deployment would require:

### Real-Time Features
- Current traffic conditions (Google Maps, Waze APIs)
- Weather data
- Special events (concerts, sports games)
- Road closures

### Route-Based Modeling
- Actual route geometry (not just straight-line distance)
- Road type (highway vs. city streets)
- Number of intersections and traffic lights
- Historical speed profiles for specific routes

### Continuous Learning
- Weekly retraining with recent data
- Online learning for rapid adaptation
- A/B testing infrastructure for model variants

### Monitoring and Alerting
- Real-time dashboards for MAE, MedAE, P90
- Segment-level monitoring
- Guardrail metrics (cancel rate, user satisfaction)

## The Experiment Design

For production deployment, we'd run an A/B test:

**Control**: Current PyTorch model (baseline)
**Treatment**: Enhanced model with:
- Real-time traffic features
- Route-based characteristics
- Segment-specific calibration

**Primary Metrics**: 
- Median Absolute Error (target: 5-10% reduction)
- P90 Absolute Error (target: 5-15% reduction)

**Guardrails**:
- Driver cancel rate (<5% increase)
- Reroute rate (<10% increase)
- User satisfaction (no degradation)

## Conclusion

Building this ETA accuracy analysis system taught us valuable lessons about:

1. **Data quality**: Proper filtering and unique identifiers are foundational
2. **Scalability**: Spark integration requires careful memory management
3. **Model robustness**: Early stopping, LR scheduling, and checkpointing are essential
4. **Production thinking**: Always consider dependencies, monitoring, and experimentation

The system successfully processed **3.5 million trips** and revealed clear patterns: short trips and peak hours are the hardest to predict accurately. These insights directly inform where to focus improvement efforts.

**Key Takeaway**: Building production-ready analytics systems requires thinking beyond just the model. Data quality, infrastructure, monitoring, and experimentation design are equally important.

---

## Code and Resources

The complete project is available with:
- Full source code (Python, SQL)
- Documentation and setup instructions
- Example outputs and visualizations

**Tech Stack**: Python 3.10+, PyTorch, Apache Spark, PostgreSQL, SQLAlchemy, Pandas, NumPy, Matplotlib

**Dataset**: NYC Taxi trip data (publicly available from NYC TLC)

---

*This project demonstrates end-to-end analytics capabilities relevant to Mapping Data Scientist roles at companies like Lyft, Uber, and Google Maps. The focus is on metric definition, segmentation, and analysis—the core skills needed to evaluate and improve ETA systems in production.*

