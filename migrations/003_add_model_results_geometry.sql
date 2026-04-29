-- 003_add_model_results_geometry.sql
-- Mirror remote model_results metadata locally and keep drawable geometry
-- on classified_image_defects for existing stats/classification flows.

CREATE TABLE IF NOT EXISTS model_results (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    img_name TEXT NOT NULL,
    class_name TEXT NOT NULL,
    confidence NUMERIC(5,4) NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    model_name TEXT,
    geometry_type TEXT,
    coordinates JSONB,
    image_width INTEGER,
    image_height INTEGER
);

CREATE UNIQUE INDEX IF NOT EXISTS model_results_ok_img_name_unique
ON model_results (img_name)
WHERE class_name = 'OK';

CREATE INDEX IF NOT EXISTS model_results_img_name_idx
ON model_results (img_name);

CREATE INDEX IF NOT EXISTS model_results_created_at_idx
ON model_results (created_at);

ALTER TABLE model_results
    ADD COLUMN IF NOT EXISTS model_name TEXT,
    ADD COLUMN IF NOT EXISTS geometry_type TEXT,
    ADD COLUMN IF NOT EXISTS coordinates JSONB,
    ADD COLUMN IF NOT EXISTS image_width INTEGER,
    ADD COLUMN IF NOT EXISTS image_height INTEGER;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'model_results_geometry_type_chk'
    ) THEN
        ALTER TABLE model_results
            ADD CONSTRAINT model_results_geometry_type_chk
            CHECK (geometry_type IS NULL OR geometry_type IN ('bbox', 'polygon', 'classification'));
    END IF;
END $$;

ALTER TABLE classified_image_defects
    ADD COLUMN IF NOT EXISTS remote_model_result_id INTEGER,
    ADD COLUMN IF NOT EXISTS model_name TEXT,
    ADD COLUMN IF NOT EXISTS geometry_type TEXT,
    ADD COLUMN IF NOT EXISTS coordinates JSONB,
    ADD COLUMN IF NOT EXISTS image_width INTEGER,
    ADD COLUMN IF NOT EXISTS image_height INTEGER;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'classified_image_defects_geometry_type_chk'
    ) THEN
        ALTER TABLE classified_image_defects
            ADD CONSTRAINT classified_image_defects_geometry_type_chk
            CHECK (geometry_type IS NULL OR geometry_type IN ('bbox', 'polygon', 'classification'));
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS classified_image_defects_remote_model_result_id_unique
ON classified_image_defects (remote_model_result_id)
WHERE remote_model_result_id IS NOT NULL;
