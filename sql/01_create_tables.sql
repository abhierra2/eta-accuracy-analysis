-- Create tables and indexes for ETA analysis
-- This script sets up the schema and derived columns for analysis

-- Note: trips_clean table is created by load_to_postgres.py
-- This script adds derived columns and indexes

-- Add derived columns if they don't exist
DO $$
BEGIN
    -- Add hour_of_day column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'trips_clean' AND column_name = 'hour_of_day'
    ) THEN
        ALTER TABLE trips_clean ADD COLUMN hour_of_day INTEGER;
    END IF;
    
    -- Add day_of_week column (0=Monday, 6=Sunday)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'trips_clean' AND column_name = 'day_of_week'
    ) THEN
        ALTER TABLE trips_clean ADD COLUMN day_of_week INTEGER;
    END IF;
    
    -- Add distance_bucket column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'trips_clean' AND column_name = 'distance_bucket'
    ) THEN
        ALTER TABLE trips_clean ADD COLUMN distance_bucket VARCHAR(10);
    END IF;
    
    -- Add trip_id column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'trips_clean' AND column_name = 'trip_id'
    ) THEN
        ALTER TABLE trips_clean ADD COLUMN trip_id BIGINT;
    END IF;
END $$;

-- Populate derived columns
UPDATE trips_clean
SET 
    hour_of_day = EXTRACT(HOUR FROM pickup_datetime),
    day_of_week = EXTRACT(DOW FROM pickup_datetime),
    distance_bucket = CASE
        WHEN haversine_distance_km < 1.609 THEN '<1mi'
        WHEN haversine_distance_km < 4.828 THEN '1-3mi'
        WHEN haversine_distance_km < 8.047 THEN '3-5mi'
        WHEN haversine_distance_km < 16.093 THEN '5-10mi'
        ELSE '10+mi'
    END
WHERE hour_of_day IS NULL OR day_of_week IS NULL OR distance_bucket IS NULL;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_trips_pickup_datetime 
ON trips_clean(pickup_datetime);

CREATE INDEX IF NOT EXISTS idx_trips_distance_bucket 
ON trips_clean(distance_bucket);

CREATE INDEX IF NOT EXISTS idx_trips_hour_of_day 
ON trips_clean(hour_of_day);

CREATE INDEX IF NOT EXISTS idx_trips_day_of_week 
ON trips_clean(day_of_week);

-- Create unique index on trip_id (if not already exists)
CREATE UNIQUE INDEX IF NOT EXISTS idx_trips_trip_id 
ON trips_clean(trip_id);

-- Create a view for training data (first 70% by pickup_datetime)
CREATE OR REPLACE VIEW trips_training AS
WITH split_point AS (
    SELECT to_timestamp(
        PERCENTILE_CONT(0.7) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM pickup_datetime))
    ) as p70_datetime
    FROM trips_clean
)
SELECT t.*
FROM trips_clean t
CROSS JOIN split_point sp
WHERE t.pickup_datetime <= sp.p70_datetime;

-- Create a view for evaluation data (remaining 30%)
CREATE OR REPLACE VIEW trips_evaluation AS
WITH split_point AS (
    SELECT to_timestamp(
        PERCENTILE_CONT(0.7) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM pickup_datetime))
    ) as p70_datetime
    FROM trips_clean
)
SELECT t.*
FROM trips_clean t
CROSS JOIN split_point sp
WHERE t.pickup_datetime > sp.p70_datetime;

-- Print summary statistics
SELECT 
    'Total trips' as metric,
    COUNT(*)::TEXT as value
FROM trips_clean
UNION ALL
SELECT 
    'Training trips',
    COUNT(*)::TEXT
FROM trips_training
UNION ALL
SELECT 
    'Evaluation trips',
    COUNT(*)::TEXT
FROM trips_evaluation;

