-- Compute ETA accuracy metrics and route quality proxy metrics
-- This script calculates median speeds from training data and applies them to evaluation data

-- Step 1: Compute median speed by hour_of_day × distance_bucket from training data
CREATE OR REPLACE VIEW median_speeds AS
SELECT 
    hour_of_day,
    distance_bucket,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY 
        haversine_distance_km / NULLIF(trip_duration_sec, 0) * 3600
    ) AS median_speed_kmh,
    COUNT(*) AS trip_count
FROM trips_training
WHERE trip_duration_sec > 0 
  AND haversine_distance_km > 0
GROUP BY hour_of_day, distance_bucket
HAVING COUNT(*) >= 10  -- Minimum trips for reliable median
ORDER BY hour_of_day, distance_bucket;

-- Step 2: Estimate ETA for evaluation trips using median speeds
-- Fallback to overall median if segment-specific median is not available
CREATE OR REPLACE VIEW trips_with_eta AS
SELECT 
    e.*,
    COALESCE(
        ms.median_speed_kmh,
        (SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY 
            haversine_distance_km / NULLIF(trip_duration_sec, 0) * 3600
        ) FROM trips_training WHERE trip_duration_sec > 0 AND haversine_distance_km > 0)
    ) AS estimated_speed_kmh,
    e.haversine_distance_km / NULLIF(
        COALESCE(
            ms.median_speed_kmh,
            (SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY 
                haversine_distance_km / NULLIF(trip_duration_sec, 0) * 3600
            ) FROM trips_training WHERE trip_duration_sec > 0 AND haversine_distance_km > 0)
        ) / 3600.0,
        0
    ) AS estimated_duration_sec
FROM trips_evaluation e
LEFT JOIN median_speeds ms 
    ON e.hour_of_day = ms.hour_of_day 
    AND e.distance_bucket = ms.distance_bucket;

-- Step 3: Compute error metrics
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

-- Step 4: Aggregate metrics overall
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

-- Step 5: Aggregate metrics by distance_bucket
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

-- Step 6: Aggregate metrics by hour_of_day
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

-- Step 7: Aggregate metrics by day_of_week
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

