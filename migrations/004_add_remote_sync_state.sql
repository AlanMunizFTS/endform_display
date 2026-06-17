-- 004_add_remote_sync_state.sql
-- Persist remote DB checkpoints locally and queue unmatched remote model results
-- so remote polling stays forward-only and efficient without mutating remote data.

CREATE TABLE IF NOT EXISTS remote_sync_state (
    source_name TEXT PRIMARY KEY,
    last_seen_id INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS remote_model_results_pending (
    remote_id INTEGER PRIMARY KEY,
    img_name TEXT NOT NULL,
    class_name TEXT NOT NULL,
    confidence NUMERIC(5,4) NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE,
    model_name TEXT,
    geometry_type TEXT,
    coordinates JSONB,
    image_width INTEGER,
    image_height INTEGER,
    first_seen_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS remote_model_results_pending_img_name_idx
ON remote_model_results_pending (img_name);

CREATE INDEX IF NOT EXISTS remote_model_results_pending_remote_id_idx
ON remote_model_results_pending (remote_id);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'remote_model_results_pending_geometry_type_chk'
    ) THEN
        ALTER TABLE remote_model_results_pending
            ADD CONSTRAINT remote_model_results_pending_geometry_type_chk
            CHECK (geometry_type IS NULL OR geometry_type IN ('bbox', 'polygon', 'classification'));
    END IF;
END $$;
